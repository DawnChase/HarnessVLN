# 一次任务如何运行

`NavigationHarness.run_task(task, stack)` 是单 Task 的生命周期边界。它只调用一次
`Agent.run(context)`，同时监听环境终止和 Agent 主动停止。

![单 Task 生命周期](/task-lifecycle.png){.architecture-asset}

## 1. 建立 ToolBus 与终止工具

Harness 先创建新的 `ToolBus`、取消事件和 terminal signal，然后注册 `nav.stop`。`nav.stop`
只负责竞争终止状态，不直接调用 Environment 的原生 Stop 动作；当前 Goal 的提交由
`nav.goal.finish` 完成。

Agent 获得的 `NavContext` 包含：

```python
NavContext(
    task=task,
    execution_id=uuid,
    tools=agent_tool_client,
    cancelled=cancel_event,
)
```

`context.nav`、`context.vln`、`context.spatial` 是对 ToolClient 的轻量类型化便捷封装。

## 2. 按资源依赖启动组件

启动顺序固定为：

1. `Environment.start(task)`：先注册 `nav.observe`、动作与 goal 工具；
2. `Memory.start(task, memory_client)`：它可依赖环境只读工具，并注册空间工具；
3. 校验 VLN requirements 对 Environment profile；
4. `VLN.start(task, vln_client)`：注册 Job 工具；
5. 校验 Agent 的 required tools 已全部存在；
6. 创建 `Agent.run`、`Environment.wait_terminal` 和 terminal wait 三个协程。

工具先注册再验证，缺失能力会在 Agent 开始决策前快速失败。

## 3. Agent 主导运行

以透传 Agent 为例，真实控制流是：

```text
Agent                  VLN navigator / worker          Environment
  | start(job)                  |                          |
  |---------------------------->|                          |
  | status(job)                 |-- observe -------------->|
  |<----------------------------|<--------------------------|
  |                             |-- move(forward) --------->|
  |                             |<--------------------------|
  | status(job)                 |        ...               |
  |<----------------------------|                          |
  | finish_goal ------------------------------------------>|
  |<---------------------------------- done / next goal ----|
  | stop(task)                  |                          |
```

Agent 轮询 Job 是控制协议，不是逐动作控制。VLN worker 在两次 status 之间可以按自己的频率执行
多次 observe/move，也可运行内部异步线程。

固定 workflow Agent 同样在一个 `run` 中执行多个步骤。自由 Agent loop 则可以在每轮选择任意
已授权工具；Harness 不需要增加另一个运行模式。

## 4. 三方终止竞争

Harness 等待以下事件中第一个完成：

| 信号 | 典型来源 | 结果 actor |
|---|---|---|
| `nav.stop` | Agent 判断任务完成或失败 | 调用者 actor |
| Environment terminal | 原生 episode 结束、动作预算、服务失败 | `environment` |
| Agent 正常返回但没 stop | Agent 实现错误 | `harness` failed |
| Agent 抛异常 | 模型/工作流错误 | `agent` failed |
| `timeout_s` | Task 超时 | `harness` failed |

Terminal signal 由锁保护，第一位 claim 者获胜。后到的 stop 仍拿到同一个 terminal，不覆盖原因。

## 5. 先封写，再回收

进入 `finally` 后，Harness 先设置 cancelled，并关闭 ToolBus 的新写调用入口。随后：

1. 停止 Environment，使原生运动不能继续；
2. 停止 VLN，取消/收敛 Job 和 worker；
3. 取消仍存活的 Agent 与环境 wait task；
4. 等待已进入的写工具调用排空；
5. 停止 Memory，原子持久化写回；
6. 读取 `Environment.result()`；
7. 把 cleanup errors 与主 terminal 分开保存。

这个顺序用于处理最危险的竞态：Task 超时或 Agent 停止时，VLN 可能仍有一个反向 move 在途。
Environment 先停止且 ToolBus 拒绝新写，使旧 worker 无法在下一 Task 中产生运动。

## 6. 返回结果

`NavigationResult` 至少包含：

- `execution_id`
- 最终 `terminal(status, reason, actor)`
- Environment 的结构化结果
- `cleanup_errors`
- 带调用序号与 outcome 的 `ToolEvent` 审计

Bench 在 Harness 返回后才评分。即使 task 失败，能够取得的清理与审计信息也保留，便于区分模型
失败、环境失败和基础设施失败。
