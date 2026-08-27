# Bench 与评分

Bench 负责定义“测什么”，Environment 负责定义“怎样执行”。每个 simulator 入口至少对应一个
真实 Bench，避免只做空壳适配器。

## 契约

```python
class Benchmark(Protocol):
    name: str
    split: str
    validation_status: str

    def cases(self) -> Iterable[BenchmarkCase]: ...
    def score(self, case: BenchmarkCase, result: NavigationResult) -> MetricSet: ...
```

`cases()` 应流式产生数据，`score()` 只使用私有 truth 与 Environment result。Runner 不理解 SR、
SPL、NE 或 GOAT 子目标指标。

## 当前矩阵

| Bench | Environment | 任务形态 | 当前数据状态 |
|---|---|---|---|
| Dummy | Dummy | 单 Goal | 内嵌 smoke |
| R2R-CE | Habitat | 语言路径跟随 | 本地已验证 |
| GOAT-Bench HM3D | Habitat + GOAT | 单 Task 多 Goal、跨目标不 reset | 本地已验证 |
| ObjectNav MP3D v1 | Habitat | 类别目标 | 本地已验证 |
| ObjectNav HM3D v2 | Habitat | 类别目标 | 本地已验证 |
| RoboTHOR ObjectNav 2021 | AI2-THOR | 类别目标 | 本地已验证 |
| VLN-PE R2R | Isaac + InternUtopia | 具身语言导航 | 专用资源缺失 |
| VLNVerse coarse | Isaac + InternUtopia | 具身语言导航 | 专用资源缺失 |

模型与 Bench 不是任意笛卡尔积。三个当前 VLN checkpoint 面向 R2R，不能因为 Environment 工具
契约相同就宣称它们能解决 ObjectNav。

## loader 的职责

- 固定 split、root、shard 与数据版本；
- 验证必要文件与最小字段；
- 生成稳定、Run 内唯一的 `case_id`；
- 把任务允许的信息映射为 `NavTask`；
- 把 reset 信息放到 `setup`；
- 把 reference path、goal 与评分常量放到 `truth`；
- 不加载整个大 split 到内存。

多 shard 数据中原生 episode id 可能重复，case id 应包含 dataset、split、scene/shard 等上下文。

## 评分的职责

Bench 从 `NavigationResult.environment` 获取执行结果，例如 trajectory、path length、native metrics、
goal distance，再与 truth 计算指标。任务失败或缺字段时应产生明确的可诊断结果，而不是静默补零。

当前 R2R 关注 SR、SPL、NE、OS；ObjectNav/GOAT 尽量保留 native measurement。官方 parity 要求
相同模拟器/数据版本、成功阈值、路径长度定义和 evaluator，而不只是指标同名。

## `validation_status`

这个字段进入 RunSummary 和 Manifest，用来表达 Bench 组合的验证等级。它不能由“此次 case 成功”
自动升级；状态是对 adapter、数据与评分链路证据的版本化判断。

## GOAT 为何是一个复合 Task

如果 Runner 把每个 GOAT goal 拆成 case，就会 reset simulator、丢失物理位置与跨目标记忆，改变
Bench 定义。当前 loader 产生一个含初始公开 Goal 的 Case，把完整 goal 序列留在 setup；
Environment 每次 `nav.goal.finish` 推进一次原生 subtask。

这也是 Harness 支持“普通 VLN 任务”和“跨任务/跨目标记忆导航”的基础：连续目标先由复合 Task
表达，真正跨 Task 的知识再由持久化 Memory 表达。
