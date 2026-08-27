# 总体分层

HarnessVLN 的架构目标不是制造一个包含所有导航范式的基类，而是让变化发生在正确边界内：
Agent 组织能力，VLN 保留完整模型，Environment 适配执行系统，Bench 适配任务与评分。

![HarnessVLN 总体架构](/architecture-overview.png){.architecture-asset}

## 六层职责

<div class="layer-stack">
  <div class="layer-row" style="--layer-color:#4a7dab"><strong>Bench / Runner</strong><p>Bench 流式产生 case 并评分；Runner 只调度完整 Task、收集有序结果。</p></div>
  <div class="layer-row" style="--layer-color:#32806a"><strong>Harness</strong><p>为一个 Task 装配组件、建立能力总线、仲裁终止并执行有界清理。</p></div>
  <div class="layer-row" style="--layer-color:#176b58"><strong>Agent Core</strong><p>唯一决策主体；固定 workflow 与自由 function-calling loop 都实现同一个 run 契约。</p></div>
  <div class="layer-row" style="--layer-color:#a55258"><strong>VLN / Memory</strong><p>作为 Agent 可委托的导航能力与空间状态能力，内部实现保持独立。</p></div>
  <div class="layer-row" style="--layer-color:#665c9e"><strong>ToolBus</strong><p>用名称、JSON Schema、actor 白名单、写属性和审计事件连接组件。</p></div>
  <div class="layer-row" style="--layer-color:#39759f"><strong>Environment</strong><p>把标准导航工具翻译为 Habitat、AI2-THOR、Isaac 或服务调用。</p></div>
</div>

`schemas/` 横跨这些层，但只保存小而稳定的数据结构：`NavTask`、`NavGoal`、`Observation`、
`Pose`、`EnvironmentTerminal` 和 `NavigationProfile`。模拟器原生类型不会进入 schema。

## 运行时依赖方向

```text
CLI -> App -> BenchRunner -> NavigationHarness
 |      |          |              |
 |      |          |              +-> Contracts + ToolBus + Schemas
 |      |          +-> Benchmark
 |      +-> ConfiguredStackFactory
 |                   |-- Agent
 |                   |-- Environment
 |                   |-- VLN
 |                   `-- Memory
 `-> Manifest
```

插件可以依赖 `harness.contracts`、`harness.runtime.NavContext`、`harness.tool_bus.Tool` 和
`schemas`；核心层不反向 import 某个具体插件。唯一的具体绑定存在于 YAML factory。

## 控制面与数据面

**控制面**由 Runner、Harness、Job 状态和 terminal signal 组成，回答“谁启动、何时停止、怎样
回收”。**数据面**由 ToolBus 调用、Observation、动作结果和 memory item 组成，回答“导航期间
传了什么”。分开后，Runner 能批量运行而无需窥探逐步观测，VLN worker 能独立进程运行而不改变
Agent API。

## 为什么不建立万能 BaseVLN

StreamVLN、JanusVLN、DualVLN 的输入历史、输出格式、推理线程和 checkpoint 加载差异很大。
Harness 只要求它们：

1. 声明导航 profile requirements；
2. 启动后注册 `vln.navigate.*` 工具；
3. Job 执行时只调用授权的导航工具；
4. 停止时使后台任务和进程静默。

这组边界足够连接，又不侵入模型内部。具体模型可以直接在进程内实现，也可继承现有 RPC
navigator 把重依赖放进 worker。

## 插件作用域

| 组件 | 支持作用域 | 原因 |
|---|---|---|
| Benchmark | task-only factory lifecycle | 一个 Run 创建一次 Bench，由它流式产生 case |
| Agent | `task` | 工作流与对话历史不能跨 case 泄漏 |
| Environment | `task` | 原生 session 和场景状态按 Task 隔离 |
| Memory | `task` 实例，可持久化文件 | 连接短寿命，知识可跨 Task |
| VLN | `task` 或 `run` | 大模型可跨 case 复用 worker/checkpoint |

run-scoped VLN 当前要求串行 case，防止同一 worker 被两个环境同时绑定。未来若 worker 明确支持
多租户，应新增资源池实现，而不是放松现有不变量。

## 扩展点不是所有内部函数

首要公开扩展点只有五个：`NavigationAgent`、`VLNNavigator`、`Environment`、`SpatialMemory`
和 `Benchmark`。Tool 名与公共 schema 是它们之间的协议。Runner、terminal arbitration、RPC
transport、Manifest writer 属于基础设施；只有出现跨插件的真实需求时才扩大接口。

继续阅读：[Task 与 Goal 模型](./task-model.md)和[插件契约总览](../extending/plugin-contract.md)。
