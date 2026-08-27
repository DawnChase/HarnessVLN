# ToolBus 与函数调用

ToolBus 是组件之间唯一的能力交换面。它既不是全局 service locator，也不是简单的函数名到 callable
字典；它同时执行输入校验、actor 授权、停止写屏障和审计。

## Tool 定义

```python
Tool(
    name="nav.move.discrete",
    description="Execute one discrete navigation action.",
    input_schema={
        "type": "object",
        "properties": {"action": {"enum": ["forward", "turn_left", "turn_right"]}},
        "required": ["action"],
        "additionalProperties": False,
    },
    handler=move,
    writes=True,
)
```

schema 使用 JSON Schema Draft 2020-12。工具注册时检查 schema，本次调用时检查 arguments。

## Actor 最小权限

Harness 为不同组件创建独立客户端：

```python
agent_client = bus.client("agent", agent.required_tools)
vln_client = bus.client("vln", vln.required_tools)
memory_client = bus.client("memory", memory.required_tools)
```

客户端只允许调用白名单中的名字，并且名字还必须已注册。VLN 通常只有 `nav.observe` 与
`nav.move.discrete`，因此不能直接调用 `nav.stop` 或读空间记忆；是否扩大权限由具体模型契约决定。

## 一次调用的路径

```text
ToolClient.call(name, arguments)
  -> check actor allowlist
  -> resolve registered Tool
  -> reject new write if bus is closing
  -> Draft202012Validator.validate(arguments)
  -> mark in-flight write when applicable
  -> await handler(actor, arguments)
  -> append ToolEvent
  -> release in-flight write
```

handler 收到 actor，Environment 可在需要时区分 Agent 与 VLN 来源。异常会记录到事件并原样向调用
方传播，不被包装成虚假的成功对象。

## 为什么区分读与写

`nav.observe`、`vln.navigate.status`、`spatial.search` 是只读；动作、Goal 提交、Task stop、Job
start/cancel 和 memory remember 是写操作。Task 终止时：

1. `close_writes()` 立即拒绝新的写调用；
2. Environment 停止；
3. VLN 停止，不再生产动作；
4. `drain_writes()` 等已进入 handler 的写调用完成。

这比只取消 Agent task 更强，因为动作可能来自独立 worker 的反向调用。

## 审计事件

ToolEvent 保存：

- 单调 `sequence`、tool `name` 与 actor；
- 截断/摘要后的 arguments；
- 调用开始时的 `monotonic_time`；
- `outcome`：ok、denied、closed、invalid 或 error；
- error 时的异常类别。

sequence 在调用进入总线时分配，事件在调用结束/拒绝时追加。并发调用时列表位置可能与 sequence
不同，应按 sequence 还原进入顺序。当前事件不记录完成时间，工具耗时需要另行 telemetry 扩展。

## 工具命名规则

当前按领域和动作分段：

- `nav.observe`
- `nav.move.discrete`
- `nav.goal.finish`
- `nav.stop`
- `vln.navigate.start/status/cancel`
- `spatial.search/remember`

新工具应表达稳定能力，而不是某个模拟器方法。例如增加连续控制可新增 `nav.move.velocity` 并声明
frame/units，而不是暴露 `habitat.step`。同名工具必须保持参数和语义向后兼容；不兼容变化应使用
新名字或显式协议版本。

## 工具不是无限扩展字典

为每个原生传感器或 SDK 方法注册一个工具会把中间件边界打穿。判断是否新增公共工具时检查：

1. 是否至少有一个 Agent/VLN 真实需要；
2. 是否能跨两个执行后端表达稳定语义；
3. frame、units、terminal 和错误语义能否写清；
4. 能否通过 profile/requirements 在运行前判断兼容；
5. 是否有 contract test 覆盖 schema 与停止竞态。
