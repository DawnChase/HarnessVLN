# 兼容与验证矩阵

兼容性按“接口成立、数据映射成立、真实链路成立、官方对照成立”分级。下面描述当前仓库证据，不把
计划能力与已验证能力混在一起。

## 模型 × 环境

| 模型 | 已接入形式 | R2R-CE + Habitat | 其他 Bench | 官方同版本 parity |
|---|---|---|---|---|
| StreamVLN | RPC v2，run-scope worker | 固定前三例真实 trace | 未声明任务兼容 | pending |
| JanusVLN Base | RPC v2，run-scope worker | 固定前三例真实 trace | 未声明任务兼容 | pending |
| InternVLA-N1-DualVLN | RPC v2，run-scope worker | 固定前三例真实 trace，专用 -30° profile | VLN-PE/VLNVerse 资源未落地 | pending |
| Dummy VLN | 进程内异步 Job | contract / dummy integration | 仅测试用途 | 不适用 |

三个真实模型固定三例指标只是 Harness 运行证据：

| 模型 | SR | SPL | NE | OS |
|---|---:|---:|---:|---:|
| StreamVLN | 1.000 | 0.904 | 1.138 | 1.000 |
| JanusVLN | 1.000 | 0.946 | 1.292 | 1.000 |
| DualVLN | 1.000 | 0.910 | 0.337 | 1.000 |

样本数只有 3，不能作为模型质量比较。trace 记录 `official_evaluator_comparison: pending`。

## Bench × Environment

| Bench | 执行后端 | 接口/数据 | 真实证据 | 当前缺口 |
|---|---|---|---|---|
| Dummy | Dummy | <span class="badge-ok">contract</span> | YAML 全闭环 | 仅用于框架测试 |
| R2R-CE | Habitat 0.3.3 | <span class="badge-ok">已落地</span> | reset、RGB-D、动作、stop；三个模型固定三例 | 完整 split 官方 parity |
| GOAT-Bench | Habitat 0.3.3 | <span class="badge-ok">已落地</span> | 连续两个 subtask、future goal 隔离、image goal render | 完整 split 与策略组合 |
| ObjectNav MP3D | Habitat 0.3.3 | <span class="badge-ok">已落地</span> | 首 shard episode 原生 SR/SPL smoke | 完整 split evaluator 对照 |
| ObjectNav HM3D | Habitat 0.3.3 | <span class="badge-ok">已落地</span> | 首 shard episode 与 scene rewrite smoke | 完整 split evaluator 对照 |
| RoboTHOR ObjectNav | AI2-THOR 2.7.2 | <span class="badge-ok">已落地</span> | Xvfb/Mesa 下真实 RGB-D、动作、Stop | 完整 validation 与策略组合 |
| VLN-PE | Isaac 4.5 + InternUtopia | <span class="badge-contract">data contract</span> | native config 生成；Isaac 基础渲染 | 专用 scene/episode/H1 policy 真实 reset |
| VLNVerse | Isaac 4.5 + InternUtopia | <span class="badge-contract">data contract</span> | native config 生成；Isaac 基础渲染 | 专用 scene/episode/H1 policy 真实 reset |

源码中的 `Benchmark.validation_status` 目前保守地使用 `contract` 或 `data_contract`；上表“真实证据”
来自额外 smoke/trace，不自动改写该字段。

## Agent × 能力

| Agent | 直接 nav | VLN Job | Memory | 复合 Goal | 自由工具选择 |
|---|---:|---:|---:|---:|---:|
| Passthrough | 否 | 是 | 否 | 是 | 否 |
| Subtask | 否 | 是，多 Job | 是 | 是 | 固定 workflow |
| 自定义 Free Agent | 可 | 可 | 可 | 由实现处理 | 接口支持，基线未实现 |

这里的“接口支持”表示 `Agent.run + ToolBus.schemas/client` 足以实现，不表示仓库已有通用 LLM Agent。

## 空间记忆

| 实现 | 查询 | 持久化 | 并发写 | 地图能力 |
|---|---|---|---|---|
| Dummy Landmark | 子串、frame、近邻、top-k | 原子 JSON，跨 Task | 串行 | 无拓扑/占据/embedding |

## 新组合的验收门槛

1. profile/requirements 静态通过；
2. task modality 与模型训练/输入契约匹配；
3. loader 无真值泄漏；
4. 真实 reset-observe-act-finish-stop；
5. timeout/cancel 后无进程、文件和运动残留；
6. 固定小样本 trace 可复核；
7. 同版本完整 split 与官方 evaluator 对照。

前两项只能证明可以尝试运行；第 7 项才支持对外声称完成 benchmark 兼容。
