# 工具目录

本页列出 v0.1 当前公共工具。实际可见工具由组件 `required_tools` 白名单决定；Environment 还会按
自己的 profile 收窄动作 enum。

## `nav.observe`

读取最新标准观测，只读。

```json
{}
```

返回：

```json
{
  "observation_id": "1",
  "source_time": 0.0,
  "received_time": 0.0,
  "frame": "FRAME",
  "channels": {},
  "pose": {"frame": "FRAME", "x": 0.0, "y": 0.0, "z": 0.0},
  "extras": {}
}
```

`pose` 可缺省；channels 由 profile 声明。RGB/Depth 在进程内可为 ndarray，经 RPC 时编码为媒体
descriptor，再由 worker 解码。

## `nav.move.discrete`

执行一个完整高层离散动作，写工具。

```json
{"action": "forward"}
```

`action` 的 enum 来自当前 Environment，例如 R2R 是 forward/turn_left/turn_right，Dual overlay
增加 stand_still。返回至少表达 accepted/action，其他计数、native tick 或反馈由 adapter 扩展。

## `nav.goal.finish`

提交当前 Goal，写工具。

```json
{"status": "completed", "reason": "model emitted STOP"}
```

`status` 是非空字符串，`reason` 可选。典型返回：

```json
{"accepted": true, "done": true}
```

复合 Task 的中间 Goal 返回：

```json
{
  "accepted": true,
  "done": false,
  "goal": {
    "goal_id": "next",
    "instruction": "find the table",
    "modality": "language",
    "public": {}
  }
}
```

该工具不结束 Task；Agent 必须根据结果继续或调用 `nav.stop`。

## `nav.stop`

结束完整 Task，写工具，由 Harness 注册且归 Agent 所有。

```json
{"status": "completed", "reason": "all navigation goals completed"}
```

返回最终获胜 terminal 的 status/reason。多个并发 stop 中第一个 claim 生效。VLN navigator 被明确
禁止把 `nav.stop` 声明为 required tool。

## `vln.navigate.start`

启动一个完整 VLN Job，写工具。

```json
{
  "instruction": "walk past the sofa and enter the kitchen",
  "options": {}
}
```

instruction 非空，options 必须是对象。返回：

```json
{"job_id": "JOB_ID"}
```

当前 RPC navigator 同时只允许一个 active Job；一个 Agent 可在前一 Job terminal 后再启动下一项。

## `vln.navigate.status`

读取 Job 状态，只读。

```json
{"job_id": "JOB_ID"}
```

返回包含 `job_id`、`state`、`reason`，实现可增加 steps 等字段。state 使用 running、succeeded、
failed、cancelled；RPC 取消过渡期内部还可能处于 cancelling，但 Agent helper 会继续读取 terminal。

## `vln.navigate.cancel`

取消 Job 而不直接结束 Task，写工具。

```json
{"job_id": "JOB_ID"}
```

重复取消应返回稳定 terminal 状态。Agent 被取消时使用 shielded cancel，确保请求不因外层取消丢失。

## `spatial.search`

查询导航地标，只读。

```json
{
  "query": "red sofa",
  "frame": "habitat_episode",
  "near_pose": [1.2, -0.4, 0.5],
  "top_k": 5
}
```

`query` 和正整数 `top_k` 必填；frame、2 至 3 维 near_pose 可选。返回 `{"items": [...]}`。

## `spatial.remember`

写入导航地标，写工具。

```json
{
  "text": "red sofa beside the corridor",
  "frame": "habitat_episode",
  "pose": [1.2, -0.4, 0.5]
}
```

text/frame 非空，pose 可选且为 2 至 3 个数。Dummy Memory 返回带 id、source_task_id 的完整 item。

## Function schema 导出

`Tool.function_schema()` 输出通用函数调用格式：

```json
{
  "type": "function",
  "function": {
    "name": "nav.observe",
    "description": "...",
    "parameters": {"type": "object", "additionalProperties": false}
  }
}
```

自由 Agent 可把 `ToolBus.schemas(allowed)` 交给模型 SDK，但实际调用仍应回到对应 ToolClient。
