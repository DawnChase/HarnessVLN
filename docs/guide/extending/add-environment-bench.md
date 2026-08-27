# 添加 Environment 与 Bench

新模拟器或真实服务通常需要一对插件：Environment 适配执行，Bench 适配任务和评分。二者可以
共享某个上游 dataset 类型，但不能合并成一个对象。

## 1. 先定义 Case 映射

```python
def cases(self):
    for raw in load_split(self.root, self.split):
        goal = NavGoal(
            goal_id=stable_goal_id(raw),
            instruction=public_instruction(raw),
            modality="language",
            public=public_goal_fields(raw),
        )
        task = NavTask(stable_task_id(raw), goal, scene_id=public_scene(raw))
        yield BenchmarkCase(
            case_id=stable_case_id(raw),
            task=task,
            setup=native_reset_fields(raw),
            truth=evaluator_fields(raw),
        )
```

先列出每个 raw 字段属于 public、setup 还是 truth，并为此写 contract test。不要在后期靠删除字段
修补信息泄漏。

## 2. 构造 Environment

Environment factory 通常接收私有 Case：

```python
def from_case(case: BenchmarkCase, **params) -> MyEnvironment:
    return MyEnvironment(case, **params)
```

构造阶段保存参数，`start(task)` 才建立原生 session、reset 并注册工具。如果 reset 失败，start
应清理已创建的部分资源后再抛出。

## 3. 声明 profile

以物理含义而不是原生名字声明：

```python
self.profile = NavigationProfile(
    observation_channels=frozenset({"rgb", "depth", "pose"}),
    motion=MotionProfile(
        tool="nav.move.discrete",
        actions=frozenset({"forward", "turn_left", "turn_right"}),
        frame="my_world",
        units="meters_degrees",
        forward_m=0.25,
        turn_deg=15.0,
    ),
    camera={"height": 480, "width": 640, "hfov_deg": 79},
)
```

若原生系统使用速度或 waypoint，不必假装是离散 0.25 m。新增对应标准工具和 profile 字段，并先
确认 Agent/VLN 真实需要。

## 4. 注册导航工具

基础环境一般返回：

- `nav.observe`：无参数，只读；
- `nav.move.discrete`：enum 来自 profile，`writes=True`；
- `nav.goal.finish`：提交原生 stop/subtask stop，`writes=True`。

所有动作 handler 使用 lock 串行原生控制，并捕获 start 时的 generation。在 stop 增加 generation；
旧调用拿锁后发现过期就抛 `ToolClosedError`。

## 5. terminal 与 result

`wait_terminal` 等待原生环境主动结束。Agent 主动结束时该 future 可以永远 pending，Harness 会取消
wait task。Environment 自身到达动作上限、服务断开或 episode terminal 时应 publish 一次。

`result` 返回 score 所需但不属于 Agent 输入的执行数据，例如：trajectory、path length、最终距离、
native metrics、动作接受状态。它在 stop 之后仍可读取。

## 6. 复合任务

若一个 Case 含多个连续 Goal：

1. Task 只暴露第一个 Goal；
2. setup 保存完整原生 goal stream；
3. Environment 维护 goal index；
4. `nav.goal.finish` 提交当前原生 subtask；
5. 返回 `done: false` 和下一个公开 Goal；
6. 最后一个 Goal 返回 `done: true`；
7. 整个过程只 start/reset 一次。

## 7. 评分与验证

Benchmark.score 从 Case truth 和 Environment result 计算。先通过 loader/score unit test，再用真实
环境做 reset-render-action-stop smoke。新增模拟器必须至少绑定一个具体 Bench；只有 generic session
mock 而没有 case/score 的入口标记为 contract，不写“已兼容”。

## 配置拆分

分别创建：

```text
config/benches/my_bench.yaml
config/envs/my_simulator.yaml
```

模型专用传感器差异另建小 overlay。不要复制整份 Environment YAML，也不要把模型名字写进共享
adapter 条件分支。
