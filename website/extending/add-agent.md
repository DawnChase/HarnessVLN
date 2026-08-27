# 添加 Agent

Agent 插件最适合承载新的工作流、自由 function-calling loop、规划器或恢复策略。它不需要修改
Runner 或 NavigationHarness。

## 1. 写最小实现

创建 `src/agents/reactive.py`：

```python
from harness.runtime import NavContext


class ReactiveAgent:
    required_tools = frozenset({
        "nav.observe",
        "nav.move.discrete",
        "nav.goal.finish",
        "nav.stop",
    })

    def __init__(self, *, max_steps: int = 100) -> None:
        self.max_steps = max_steps

    async def run(self, context: NavContext) -> None:
        for _ in range(self.max_steps):
            if context.cancelled.is_set():
                return
            observation = await context.nav.observe()
            action = self.choose_action(observation)
            if action == "stop":
                transition = await context.nav.finish_goal("completed")
                if transition["done"]:
                    await context.nav.stop("completed", "all goals completed")
                    return
                continue
            await context.nav.move_discrete(action)
        await context.nav.stop("failed", "agent step budget reached")

    def choose_action(self, observation: dict) -> str:
        ...
```

这是 Agent 直接使用传统导航工具的形式。它与 VLN 透传形式平级，证明 Agent 不是 VLN 的薄壳。

## 2. 调用 VLN Job

不要在 Agent 中复制 status/cancel 竞态，可复用：

```python
from agents._jobs import run_vln_job

status = await run_vln_job(context, instruction, poll_period_s=0.1)
if status["state"] != "succeeded":
    await context.nav.stop("failed", status.get("reason", "VLN failed"))
    return
```

一个 Agent 可按 workflow 启动多个 Job。Job 之间是否重新观察、查 Memory 或改写指令由 Agent 决定。

## 3. 接入通用 LLM function calling

将模型供应商响应先转换为内部 `{name, arguments}`，再调用 `context.tools.call`。循环至少维护：

- 当前公开 Goal；
- 已调用工具与紧凑结果；
- 活跃 VLN Job id；
- 最大轮数/时间预算；
- schema/tool error 的可恢复策略；
- terminal/取消状态。

不要把 Tool handler 或 Environment 原生对象直接交给 LLM SDK。可用工具清单必须从 Agent 的允许集
产生，结果仍经 ToolBus。

## 4. 配置 factory

```yaml
stack:
  agent:
    factory: agents.reactive:ReactiveAgent
    params:
      max_steps: 100
```

Agent 只支持 task scope；每个 case 创建新实例。

## 5. 处理复合 Goal

`finish_goal` 返回 `done: false` 时，结果中包含下一个公开 Goal。便捷 `NavContext.task` 是启动时的
不可变对象，不会自动改写；Agent 应把返回 Goal 保存进自身 runtime state，或至少使用返回的
instruction。参考 `PassthroughVLNAgent` 的循环。

## 6. 测试清单

- required tool 缺失时在 run 前失败；
- 正常完成一定调用 stop；
- run 正常返回但没 stop 被 Harness 标成失败；
- 超预算与 policy 异常产生明确 reason；
- 取消活动 Job 后无后台 task；
- compound task 不 reset 且按返回顺序处理 Goal；
- 禁止读取 `BenchmarkCase.setup/truth`；
- 使用 Memory 时覆盖无结果、frame 不同和 write failure。

## 常见错误

把 `Agent.run` 写成“一次观察返回一次动作”会重新引入 Runner 主导范式；把分解器、模型加载与
模拟器控制全部塞入一个 Agent 又会破坏边界。Agent 负责**编排**，每项专用能力仍通过插件工具调用。
