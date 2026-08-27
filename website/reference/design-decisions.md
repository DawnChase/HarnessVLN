# 设计取舍

本页记录当前架构刻意做出的选择，以及暂时没有引入的复杂度。它帮助后续扩展判断某个限制是遗漏，
还是为了保护边界设置的不变量。

## Agent 主导，而非 Runner 主导

<div class="decision-grid">
  <div class="decision-card"><strong>选择</strong><p>Runner 只启动完整 Task；每个 Task 只调用一次 Agent.run。</p></div>
  <div class="decision-card"><strong>代价</strong><p>Agent 必须正确处理循环、预算、Goal 与 stop，不能依赖 Runner 兜底逐步推进。</p></div>
</div>

原因是不同 VLN 有不同内部频率与状态。逐步 Runner 会强迫所有模型降格为一步 policy，并阻碍自由
function-calling Agent。Harness 仍负责 timeout 和生命周期，但不夺取决策权。

## Agent 与 VLN 分开

Agent 组织工具与工作流；VLN 保留完整模型导航循环。透传 Agent 很薄，但它仍处理 Job terminal、
复合 Goal 与 Task stop。这样同一 VLN 可被固定 workflow 或自由 Agent 使用，同一 Agent 也可换模型。

## Bench 与 Environment 分开

Bench 持有 loader/truth/score，Environment 持有 native session/action/result。共享 simulator 的 R2R、
ObjectNav 与 GOAT 可复用中间件，同时保持各自数据与评分语义。新 simulator adapter 必须绑定具体
Bench 才能谈兼容。

## 小 Protocol + 可扩展映射

公共数据类只固定身份、指令、时间、坐标框架和 observation envelope；模态 channel 与 plugin
requirement 使用映射。这样 v0.1 能扩展而不预先设计万能 schema。代价是扩展字段需要文档、版本与
contract test，不能任意堆键。

## ToolBus，而非直接对象引用

ToolBus 让能力可发现、可校验、可授权、可审计，并能在停止时封写。额外开销是 JSON Schema 校验
和显式工具定义；导航仿真/模型推理的成本远高于这部分，换来的竞态边界值得保留。

## VLN Job，而非一步 `predict`

Job 允许模型自主多次 observe/move，适配视频模型、异步大小脑和连续轨迹。Agent 通过 start/status/
cancel 管理委托。代价是必须认真处理 Job 生命周期、反向调用和媒体释放。

## task-scope 默认，VLN 可 run-scope

Agent/Environment 按 Task 隔离，避免状态泄漏。大模型 checkpoint 加载昂贵，允许 VLN worker 跨
case 复用，但当前强制串行并在任务间 detach tools/release media。若需要并行，下一步是 worker
pool，而非共享一个 navigator。

## Memory 实例短寿命，存储可长寿

Task-scope 实例使 ToolClient 和任务上下文清晰；原子文件/数据库使知识跨 Task。基础 Dummy Memory
只验证边界，不提前引入向量库、地图服务和 evolution pipeline。

## 当前不做自进化层

基础版本聚焦导航闭环、空间记忆和兼容矩阵。记忆压缩、经验提炼、策略更新可能分别属于存储维护、
Agent workflow 或训练系统，过早合成一个 `evolution` 插件会隐藏触发条件和评测污染风险。真实研究
需求明确后，再按输入、输出、时机、可见性与 rollback 设计独立协议。

## YAML factory，而非安装式插件注册

项目无需 package install 或中央 registry。`PYTHONPATH=src` 配合 `module:object` factory 足以快速
迭代；不同模型仍按子目录隔离。代价是结构类型错误在加载/启动期发现，因此配置与 contract test
必须严格。

## 一个 Conda 环境，进程隔离依赖

首发追求一次环境创建能运行 Agent、模型与模拟器，避免用多个 Python 环境掩盖版本冲突。worker
进程用于 CUDA/生命周期隔离，不用于切换依赖集合。若未来两个后端存在不可调和 ABI，服务边界应
显式版本化，而不是让 Harness API 感知环境管理细节。

## 明确保守的兼容声明

adapter、data contract、真实 smoke、固定 trace 和官方 parity 分开记录。保守声明会让矩阵看起来
进展更慢，但它防止接口存在被误读为指标可信，是科研 Harness 必须保留的证据标准。
