# Agent Core

Agent 是 HarnessVLN 的主体。公共契约刻意只有一个入口：

```python
class NavigationAgent(Protocol):
    required_tools: frozenset[str]

    async def run(self, context: NavContext) -> None: ...
```

一个 Task 只调用一次 `run`。固定流程和自由循环的差异完全留在这个方法内部，不需要两套 Runner。

## `NavContext` 提供什么

| 字段/属性 | 用途 |
|---|---|
| `task` | 当前公开 `NavTask`；复合任务开始时只含第一个 Goal |
| `execution_id` | 本次执行身份，可用于日志关联 |
| `tools` | 受 actor 白名单限制的原始 `ToolClient` |
| `cancelled` | Harness 开始终止时置位的 `asyncio.Event` |
| `nav` | `observe / move_discrete / finish_goal / stop` 便捷接口 |
| `vln` | `start / status / cancel` Job 接口 |
| `spatial` | `search / remember` 空间记忆接口 |

便捷接口不绕过 ToolBus；所有调用仍会权限校验、schema 校验和审计。

## 当前 Agent 1：透传 VLN

`PassthroughVLNAgent` 代表“模型本身就是完整导航策略”的最短路径：

```text
current instruction
  -> vln.navigate.start
  -> poll status until terminal
  -> nav.goal.finish
      | done=false -> use returned next instruction, repeat
      ` done=true  -> nav.stop(completed)
```

它不获取 Observation，也不要求 VLN 每次只输出一个动作。观察与动作由 VLN Job 自主反向调用。
因此 StreamVLN、JanusVLN、DualVLN 的内部范式都可保持原样。

## 当前 Agent 2：子任务工作流

`SubtaskNavigationAgent` 展示固定 workflow 插件：

1. `SentenceDecomposer` 按 `then`、句点、分号拆分指令；
2. 每个步骤先查询 `spatial.search(top_k=3)`；
3. 为每一步运行一个完整 VLN Job；
4. 成功后写入 `spatial.remember`；
5. 所有步骤结束后提交当前 Goal。

默认 decomposer 是确定性 baseline，也可注入同步或异步 callable，例如 LLM planner。当前实现没有
把检索结果自动拼进 VLN prompt；它只证明规划、记忆和模型委托的接口可组合，避免把尚未实现的
策略描述为有效算法。

## 自由 Agent loop 如何接入

自由 Agent 同样实现 `run`，在循环中根据状态选择工具：

```python
class FreeNavigationAgent:
    required_tools = frozenset({
        "nav.observe", "nav.move.discrete", "nav.goal.finish", "nav.stop",
        "vln.navigate.start", "vln.navigate.status", "vln.navigate.cancel",
        "spatial.search", "spatial.remember",
    })

    async def run(self, context: NavContext) -> None:
        state = {"task": context.task, "history": []}
        while not context.cancelled.is_set():
            call = await self.policy.next_call(state)
            result = await context.tools.call(call.name, call.arguments)
            state["history"].append((call, result))
            if call.name == "nav.stop":
                return
```

这只是控制结构；真正实现应限制最大轮数、处理工具错误、取消在途 VLN Job，并确保所有正常终点
都显式调用 `nav.stop`。

## Agent 的责任

- 只声明实际会调用的 `required_tools`；
- 解释 VLN Job terminal，而不是把 `succeeded` 直接等同整 Task 成功；
- 在复合任务中处理 `nav.goal.finish` 返回的新 Goal；
- 收到取消或自身取消时关闭 VLN Job；
- 正常返回前调用 `nav.stop`；
- 把 workflow 状态保留在实例内，不写全局变量。

Agent 不负责加载 dataset、reset 环境、计算指标、结束原生进程或写 Manifest。

## 选择哪一种 Agent

| 场景 | 建议 |
|---|---|
| 评测一个端到端 VLN 原始能力 | `PassthroughVLNAgent` |
| 验证“规划 → 模型 → 记忆”组合 | `SubtaskNavigationAgent` |
| 研究动态工具选择、反思或恢复 | 新增自由 loop Agent 插件 |
| 复现论文固定阶段流程 | 把阶段明确写进一个 workflow Agent |

不要把特定论文 workflow 塞进 Harness runtime；它应与透传 Agent 一样是可替换插件。
