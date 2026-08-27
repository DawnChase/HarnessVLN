# Environment 中间件

Environment 是标准导航工具与原生执行系统之间的唯一边界。Agent 和 VLN 不持有 Habitat Env、
AI2-THOR Controller、Isaac Env 或真机 SDK 对象。

## 契约

```python
class Environment(Protocol):
    profile: NavigationProfile

    async def start(self, task: NavTask) -> Sequence[Tool]: ...
    async def stop(self, reason: str) -> None: ...
    async def wait_terminal(self) -> EnvironmentTerminal: ...
    def result(self) -> dict[str, Any]: ...
```

一个实例只服务一个 Task。factory 可以额外接收私有 `BenchmarkCase`，用来 reset 正确 episode。

## Profile 是组合契约

`NavigationProfile` 描述环境**实际**提供的能力：

```python
NavigationProfile(
    observation_channels=frozenset({"rgb", "depth", "gps", "compass", "pose"}),
    motion=MotionProfile(
        tool="nav.move.discrete",
        actions=frozenset({"forward", "turn_left", "turn_right"}),
        frame="habitat_episode",
        units="meters_degrees",
        forward_m=0.25,
        turn_deg=15.0,
    ),
    camera={"height": 480, "width": 640, "hfov_deg": 79},
)
```

它不是运行时观测；它是启动阶段检查 VLN 与环境是否能组合的能力声明。

## Habitat

共享 Habitat adapter 负责：

- 从 Bench case 创建官方 dataset episode 并 reset；
- 把 RGB、Depth、GPS、Compass 和可选 ObjectGoal 归一到 Observation；
- 映射离散动作和原生 stop / subtask stop；
- 保存 trajectory、path length、最小 geodesic distance 与 native metrics；
- 用 generation fence 阻止停止后的陈旧 move。

R2R-CE、ObjectNav MP3D/HM3D 共用 adapter，但使用独立 Bench 和 YAML。GOAT 在其上扩展复合
Goal：未来目标私有保存，`nav.goal.finish` 执行 `subtask_stop` 并揭示下一个公开 Goal。

## RoboTHOR / AI2-THOR

`RoboTHOREnvironment` 包装固定 Unity build 的 Controller：reset scene、TeleportFull 到起点，
将公共动作映射为 `MoveAhead`、`RotateLeft/Right`、`LookUp/Down`。Stop 时按目标物体可见性形成
success，result 保存动作反馈、轨迹和路径长度。

默认 profile 不公开 pose 与原生 action feedback，避免不符合任务设定的信息泄漏；可由配置显式
开启。被障碍阻挡的 Move 是动作结果，不自动变成 Harness 异常。

## Isaac / InternUtopia

`IsaacNavigationEnvironment` 是高层单车道 adapter。一个公共 `nav.move.discrete` 会反复推进
native tick，直到 observation 标记高层动作完成、episode terminal，或超过
`max_native_ticks_per_action`。

VLN-PE 与 VLNVerse 各有入口，负责坐标系、episode 生成和上游 extension 差异；底层复用同一
Isaac adapter。SimulationApp 是进程级资源，因此配置 `serial: true`。

当前专用场景、H1 USD/locomotion policy 与 episode 未完整落地，两条组合的状态是
`data_contract`，不是实机 episode 验收。

## 未来真机服务

真机 adapter 仍实现相同 Contract，但应把这些细节留在内部：认证与重连、传感器时间同步、
坐标变换、动作 ACK、急停、速度限幅和服务健康检查。ToolBus 的写屏障是进程内保障，不替代机器人
控制服务自己的安全状态机。

## Environment 的停止要求

`stop(reason)` 必须：

1. 可在 start 部分失败后调用；
2. 幂等；
3. 在返回前阻断新的原生运动；
4. 使陈旧并发调用通过 generation/token fence 失败；
5. 有界关闭 session、controller 或服务连接；
6. 不因为评分逻辑而延迟资源释放。

`result()` 在停止后调用，应返回可 JSON 化或可由 Manifest serializer 降级处理的数据。
