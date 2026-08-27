# 并行与生命周期

HarnessVLN 的默认并行单位是**完整 Task**，不是 Task 内单步。并行策略同时受组件作用域、原生
资源限制和跨 Task 写入影响。

## Runner 的有界并行

`BenchRunner` 使用一个容量为 `parallelism * 2` 的队列：producer 流式读取 `benchmark.cases()`，
多个 consumer 各自构造完整 stack 并调用 `run_task`。结果可能乱序完成，最后按输入 index 排序。

```text
Benchmark.cases() -> bounded queue -> worker 0 -> whole Task A
                                  -> worker 1 -> whole Task B
                                  -> worker 2 -> whole Task C
```

该设计不会一次把百万级 split 全部装入内存，`max_cases` 也在 producer 侧截断。

## Task 内有哪些并发

一个 Task 内存在协程并发，但控制权仍属于 Agent：

- `Agent.run` 与 `Environment.wait_terminal` 同时等待；
- VLN Job 可以作为后台协程或独立 worker 进程运行；
- RPC worker 可发出异步反向 tool call；
- ToolBus 允许只读调用并行，并追踪在途写调用；
- Memory 由自身实现决定是否并行查询。

“Agent 与 VLN 同时运行”不意味着 Runner 分别调度它们。VLN 的生命周期由 navigator 管理，
Agent 通过 Job 工具委托和观察状态。

## 作用域与实例数

假设一个 Run 有 100 个 case：

| 配置 | Agent 实例 | Env 实例 | VLN 实例/worker | Memory 实例 |
|---|---:|---:|---:|---:|
| 全部 task-scope | 100 | 100 | 100 | 100 |
| VLN `scope: run` | 100 | 100 | 1 | 100 |

run-scoped VLN 在 case 之间解除旧 ToolClient 绑定、释放 Job media，再绑定新环境；checkpoint 和
worker PID 保持不变。Run 结束后 `ConfiguredStackFactory.close_run()` 负责关闭。

## 何时必须串行

StackFactory 的 `requires_serial` 在以下任一条件成立时为真：

- 某组件配置 `serial: true`；
- VLN 使用 `scope: run`；
- Memory 存在且 `writeback` 默认为 true。

此时 `runner.parallelism > 1` 会在运行前报错，而不是带着共享状态冒险执行。Isaac Kit 是典型的
进程级串行资源；Dummy Landmark Memory 的单文件写回也默认串行。

::: tip 扩展并行的正确方式
需要多个 Isaac case 并行时，使用多个隔离服务进程并让每个 Environment 独占连接；需要共享
Memory 并行写时，实现事务型存储。不要只删除 `serial` 标记。
:::

## 多频 VLN 如何保留

DualVLN 一类大小脑不同频模型不由 Harness 拆开。一个 Job 在 worker 内维护 System 1、System 2、
帧窗口与连续轨迹；worker 可在需要时主动 `nav.observe`，推理后调用多个动作或 `stand_still`。
外部只看到 Job 状态与审计事件。

这意味着时间尺度分三层：

1. Runner 的 Task 调度频率；
2. Agent 的 workflow / Job 轮询频率；
3. VLN 内部推理与原生 Environment tick 频率。

三层互不假定一一对应。Isaac 的一个 `nav.move.discrete` 还可能在中间件中推进最多数千 native tick，
直到高层动作完成。

## 取消不是简单 `task.cancel()`

取消时可能存在 GPU 进程、子进程组、文件映射和在途运动。现有实现使用：

- shielded cancel/close，确保调用者被取消时清理仍能收敛；
- worker process group 的 TERM/KILL 兜底与同步 fence；
- Job seal，拒绝已关闭 Job 的新反向调用；
- media release，按 Job 清理 file-backed array；
- ToolBus write fence 与 drain；
- 独立记录 cleanup error，不覆盖主异常。

编写新插件时，`stop()` 必须幂等、可在部分启动状态调用，并在返回前阻断后续副作用。
