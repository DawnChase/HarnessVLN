# HarnessVLN 基础架构

HarnessVLN 是 Agent 主导的导航 Harness。Runner 只并发提交完整 Task 并评分；每个 Task 只调用一次 `Agent.run(scope)`，不 observe、act 或 step。

## 核心组件

- `NavigationAgent`：拥有循环，主动调用 VLN、导航工具与 Memory，最终调用 `nav.stop`。
- `TaskSession`：只暴露当前 `NavGoal`。普通 VLN Task 只有一个 Goal；GOAT Task 连续提供 5–10 个 Goal。
- `Environment`：独占模拟器/真机控制，统一观测信封、超时与 stop fencing；原生类型留在 adapter 内。
- `VLNNavigator`：封装完整 VLN，以可取消 Job 保留模型自己的循环、状态和频率。
- `SpatialMemory`：独立插件；Dummy 保存 landmark，可按 Task 或共享目录持久化。
- `Benchmark`：加载公开 Task，保管私有真值并评分。

## 生命周期

Harness 打开 Env、Memory、VLN 后创建 Agent，typed facade 与 function calling 共用 ToolBus。GOAT 中 Agent 以 `nav.goal.finish` 推进目标，但不重置位姿、Agent、VLN session 或 Memory，最后才 `nav.stop`。步数与 terminal 由各 Bench/Environment 按原生规则处理，Runner 不注入动作。

Habitat、AI2-THOR、两个 Isaac 入口各自映射。Isaac adapter 把 warmup 和多个 physics tick 收敛为一次高层动作；VLN-PE 与 VLNVerse 保留不同 factory。普通 Task 可并行，共享写回 Memory 时串行；v0.1 的两个 Isaac 入口均限制为单 lane。

所有进程统一从 Python 3.10 的 Conda 环境 `harnessvln` 启动。环境与模型的独立进程用于
故障、生命周期和 GPU 资源隔离，不代表维护多套互相漂移的 Python 环境。

```text
src/
  harness/  agents/  envs/  vln/  memory/  benches/
```

`harness/` 不依赖具体插件；新增实现只修改所属目录、配置和测试。兼容矩阵见 [v0.1-target.md](v0.1-target.md)。

当前 Habitat、THOR、Isaac 均完成 mock contract；本机未安装对应 runtime，因此真实 reset/render/physics 与官方 metric parity 仍是 release gate，不能标记为 native verified。
