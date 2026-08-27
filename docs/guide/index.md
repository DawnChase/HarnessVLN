---
title: HarnessVLN
description: Agent 主导的模块化视觉语言导航 Harness
---

<div class="doc-masthead">
  <p class="eyebrow">Agent-led Navigation Harness</p>
  <h1>HarnessVLN</h1>
  <p class="lead">一个以 Agent 为唯一决策主体、以类型化工具连接完整 VLN 模型、空间记忆、Bench 与不同模拟器的导航实验基座。</p>
</div>

<div class="status-strip">
  <div><strong>Agent-led</strong><span>Runner 不接管逐步控制</span></div>
  <div><strong>4 类插件</strong><span>Agent / VLN / Env / Memory</span></div>
  <div><strong>3 个 VLN</strong><span>Stream · Janus · Dual</span></div>
  <div><strong>147 tests</strong><span>当前提交的全量回归</span></div>
</div>

## 先看一张图

![HarnessVLN Agent 主导架构](/architecture-overview.png){.architecture-asset}

这张图只表达一个核心事实：**Runner 启动完整 Task，Agent 决定 Task 内发生什么。** Agent
可以直接调用传统导航工具，也可以启动一个保留内部状态和频率的 VLN Job；VLN Job 又能通过
同一个 ToolBus 主动观察与移动。Environment 始终独占模拟器或真机的原生控制权。

::: tip 建议阅读顺序
第一次阅读按“心智模型 → 十分钟运行 → 总体分层 → 一次任务如何运行”前进。准备开发插件时，
再进入“核心组件”和“扩展开发”；定位实验问题时直接查“结果与 Manifest”和“故障定位”。
:::

## 按你的目标进入

<div class="route-grid">
  <a class="route-card" href="./getting-started/mental-model.html"><strong>我想先理解项目</strong><p>从 Runner、Agent、VLN、Environment 的职责边界开始，不先陷入代码细节。</p></a>
  <a class="route-card" href="./getting-started/quick-start.html"><strong>我想先跑起来</strong><p>运行零外部依赖的 dummy 链路，再切换到真实 Habitat + R2R。</p></a>
  <a class="route-card" href="./architecture/execution-flow.html"><strong>我想理解一次 Task</strong><p>沿启动、工具注册、终止竞争、写屏障和逆序清理逐步展开。</p></a>
  <a class="route-card" href="./extending/plugin-contract.html"><strong>我要增加新模块</strong><p>先选正确的插件边界，再按最小 Protocol、requirements 和 YAML 接入。</p></a>
  <a class="route-card" href="./usage/configuration.html"><strong>我要组合实验</strong><p>了解多个 YAML 如何深度叠加，以及 factory、scope、serial 的真实语义。</p></a>
  <a class="route-card" href="./reference/compatibility.html"><strong>我要判断兼容状态</strong><p>区分 contract、真实 smoke、固定三例 trace 与官方 evaluator parity。</p></a>
</div>

## 项目的五条设计约束

1. **Agent 拥有循环。** 每个 Task 只调用一次 `Agent.run(context)`；何时观察、行动、调用 VLN、切 Goal 和停止都由 Agent 决定。
2. **VLN 不被拆开。** 大小脑、多频线程、KV cache、历史帧和轨迹状态继续属于模型插件内部。
3. **Environment 是中间件。** 它把 Habitat、AI2-THOR、Isaac 或未来真机服务映射为稳定工具，而不是泄露原生对象。
4. **Bench 保管真值。** Agent 只看公开 `NavTask`；起点、原生 episode 和评分真值留在 `BenchmarkCase` 私有字段。
5. **并行发生在完整 Task。** Runner 可以并行多个完整 Task，但不会并行一个 Task 内的 observe-act 步骤。

## 当前实现范围

| 维度 | 已实现 | 仍在发布门禁中 |
|---|---|---|
| Agent | 透传 VLN、子任务拆分 + Memory | 动态 DAG、多 Agent、在线学习 |
| VLN | StreamVLN、JanusVLN、DualVLN 独立子目录与 RPC worker | 同版本官方 evaluator parity |
| Environment | Habitat、RoboTHOR、Isaac/InternUtopia 入口 | VLN-PE/VLNVerse 大型专用资源真实 episode |
| Bench | R2R-CE、GOAT、ObjectNav、RoboTHOR、VLN-PE、VLNVerse | 五条组合的完整 validation split release gate |
| Memory | 可跨 Task 持久化的 Dummy Landmark Memory | 拓扑图、稠密地图、embedding 检索 |

::: warning “有 adapter”不等于“已完整兼容”
本指南会明确标出 `contract`、`data_contract`、真实 smoke 和 release gate。缺少官方对照或大型
资源的组合不会因为接口存在就被描述为已经验收。
:::

## 文档与构建

Markdown 源文件位于 `docs/guide/`，VitePress 生产构建输出到仓库根目录 `page/`：

```bash
cd docs/guide
npm install
npm run docs:build
npm run docs:preview
```

继续阅读：[先建立正确心智模型](./getting-started/mental-model.md)。
