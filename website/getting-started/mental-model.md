# 先建立正确心智模型

HarnessVLN 不是把每个模拟器包装成 `reset / observe / act` 后再由 Runner 逐步驱动的评测脚本。
它把**完整导航任务的控制权交给 Agent**，Runner 只负责准备和调度任务。

## 从两种循环的差别开始

传统 Runner 主导循环通常是：

```text
Runner: reset -> observe -> Agent.predict -> env.step -> observe -> ... -> score
```

这种结构默认“每次观测只产生一次动作”，会把 VLN 模型自己的视频缓存、异步大小脑、连续轨迹
或多频控制拆散。HarnessVLN 的调用关系是：

```text
Runner -> NavigationHarness.run_task(task, stack)
                         |
                         +-> Agent.run(context)       # 整个 Task 只调用一次
                               |-- nav.observe
                               |-- nav.move.discrete
                               |-- vln.navigate.start
                               |       `-- VLN 内部反向调用 observe / move
                               |-- spatial.search / remember
                               |-- nav.goal.finish
                               `-- nav.stop
```

Agent 可以写成固定 workflow，也可以是完全自由的函数调用循环。Harness 只规定生命周期和工具
边界，不规定 `Agent.run` 内部要执行多少轮、每轮调用什么。

## 四个容易混淆的角色

<div class="concept-grid">
  <div class="concept-card"><strong>Runner</strong><p>遍历 Bench case、限制完整 Task 并发数、收集评分并输出 Manifest。它不读取观测，也不产生动作。</p></div>
  <div class="concept-card"><strong>NavigationHarness</strong><p>装配一个 Task 的组件，建立 ToolBus，处理终止竞争，并保证资源按顺序回收。</p></div>
  <div class="concept-card"><strong>Agent Core</strong><p>唯一决策主体。它决定直接行动、委托 VLN、查询记忆、切换子目标或终止任务。</p></div>
  <div class="concept-card"><strong>VLN plugin</strong><p>一套完整导航模型。模型内部的状态、推理频率、线程和轨迹策略留在插件内部。</p></div>
</div>

Environment 与 Bench 也必须分开理解：Environment 管原生仿真/真机执行，Bench 管 case 加载、
私有真值和评分。二者恰好都与某个数据集有关，但职责不同。

## Agent 与 VLN 不是同一个抽象

- **Agent** 回答“这项任务现在应该组织哪些能力”。例如先查空间记忆、拆成两段、调用 VLN，
  失败后重新观察，最后停止。
- **VLN** 回答“给定一条导航指令，模型内部如何完成导航”。例如 StreamVLN 保留帧历史，
  DualVLN 保留 System 1 / System 2 的异步节奏。
- Agent 可以完全不接 VLN，只用 `nav.*` 工具；也可以把指令原样交给一个 VLN Job。
- 同一 VLN 无需知道上层使用透传 Agent 还是工作流 Agent。

因此二者在 Task 运行期可以同时存活、异步推进，但它们不是“两个平级 Agent”。Agent 启动和
管理 VLN Job；VLN Job 获得受限的反向工具客户端，自主观察和行动。

## 三层稳定边界

1. **公开任务边界**：Agent 只拿到 `NavTask`，未来 Goal 和评分真值不进入上下文。
2. **工具边界**：组件只能调用 `required_tools` 声明过的工具，参数逐次经过 JSON Schema 校验。
3. **原生环境边界**：Habitat、AI2-THOR、Isaac 对象只存在于 Environment 内部。

这三层使新模型、新模拟器和新 Agent 能独立变化。兼容性不是靠巨大的统一观测字典，而是靠
`NavigationProfile` 在启动前验证当前 VLN 所需的通道、动作和相机参数。

## 一句话判断模块该放哪里

| 问题 | 归属 |
|---|---|
| 怎样遍历 split、读取 episode、算 SPL？ | Bench |
| 怎样把 `forward` 映射为 Habitat action id？ | Environment |
| 怎样把一条指令拆成多个导航段？ | Agent |
| 怎样维护图像历史并预测动作？ | VLN |
| 怎样检索跨 Task 的地标？ | Memory |
| 怎样校验调用、封住停止后的运动？ | ToolBus / Harness |
| 怎样限制同时运行几个完整 case？ | Runner |

下一步：[十分钟运行](./quick-start.md)。
