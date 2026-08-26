# HarnessVLN 基础架构

HarnessVLN 是 Agent 主导的导航基座。Runner 只创建完整 Task、并行调度和汇总评分；每个
Task 只调用一次 `Agent.run`，不执行固定的 observe-act 循环。

二级目录为 `harness/ agents/ benches/ envs/ vln/ memory/ schemas/`，不再增加安装包层。

- `harness`：生命周期、配置、ToolBus、能力校验和整任务 Runner，不依赖具体插件。
- `agents`：可替换控制内核；透传、固定工作流和自由循环共用 `Agent.run(context)`。
- `envs`：模拟器/真机中间件，独占原生控制权，把观测、动作屏障、终止和指标映射为导航工具。
- `vln`：完整模型插件；每个模型独占子目录，以可取消 Job 保留内部状态与频率。
- `memory`：导航空间记忆插件；基础实现持久化 landmark，支持跨 Task 查询。
- `benches`：加载公开 Task，保管私有真值并调用对应评分器。

启动顺序为 Env -> Memory -> VLN -> Agent。Agent 主动调用 `nav.observe`、VLN、移动、
`nav.goal.finish` 和 `nav.stop`；ToolBus 统一类型校验、权限、审计与停止写屏障。GOAT 的多个
Goal 共用同一 Env、Agent、VLN 和 Memory，切换 Goal 不 reset。

Runner 只在完整 Task 级并行。普通无共享状态组件可并发；共享写回 Memory 和进程内唯一的
Isaac SimulationApp 由组件 `serial` 元数据约束。未来多 GPU Isaac 并行使用多个隔离环境
服务进程，不改变 Agent 接口。

组件由 YAML 的 `module:factory` 组合；接口固定最小生命周期，工具和 requirements 数据可扩展。
兼容矩阵与验证状态见 [v0.1-target.md](v0.1-target.md)。
