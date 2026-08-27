# 术语与边界

本页给出后续文档使用的精确定义。先区分这些对象，可以避免把“任务”“目标”“episode”“一次模型
推理”和“一次评测运行”混为一谈。

## 核心术语

| 术语 | 在 HarnessVLN 中的含义 |
|---|---|
| Run | 一次 CLI / API 批量运行；包含一个 Bench split 的若干 Task |
| Case | Bench 内部对象，含公开 Task、私有 setup 与评分 truth |
| Task | 交给 Agent 的一次完整导航会话；可能包含连续多个 Goal |
| Goal | 当前公开的导航目标，含 instruction、modality 和 public 元数据 |
| Job | Agent 委托给 VLN 的一次导航请求；一个 Task 可启动多个 Job |
| Tool call | 组件经 ToolBus 发起的一次类型化能力调用 |
| Native step | 模拟器或真机的一次底层推进；不一定等于一个导航动作 |
| Record | 一个 Case 的 Task 结果、指标、错误和审计集合 |
| Manifest | 一次 Run 的可复现实验记录，原子写入 JSON |

## Task 不总等于单目标 episode

普通 R2R case 通常是一个 Task 对应一个 Goal。GOAT 则把一个含 5 至 10 个目标的复合 episode
表示成**一个 Task**：Agent 只看到当前 Goal，调用 `nav.goal.finish` 后 Environment 才揭示下一个。
模拟器 session 和 Task 级空间记忆不 reset。这正是终身/跨目标导航与“把每个 goal 当独立 case”
的关键区别。

## 公开数据与私有数据

```text
BenchmarkCase
  task: NavTask              -> Agent 可见
    task_id
    scene_id
    goal: NavGoal
      goal_id
      instruction
      modality
      public
    public
  setup: Mapping             -> Environment 可用，Agent 不可见
  truth: Mapping             -> Benchmark.score 可用，Agent 不可见
```

起始位姿、原生 episode、最短路径和最终答案等不应塞进 `NavTask.public`。公开字段应只包含任务
协议明确允许 Agent 使用的信息。

## 状态的所有权

| 状态 | 所有者 | 默认寿命 |
|---|---|---|
| Agent workflow / 对话历史 | Agent | Task |
| 帧历史、KV cache、大小脑线程 | VLN | Job 或 run，取决于实现 |
| 仿真器 session、机器人连接 | Environment | Task |
| 当前与未来复合 Goal | Environment / Bench | Task |
| Landmark 内存文件 | Memory | 可跨 Task / Run |
| case 序号与结果顺序 | Runner | Run |

Harness 不把某个模块的内部状态提升为公共对象。需要跨边界的信息必须经过显式 schema 或工具。

## `terminal`、`goal.finish` 与 `stop`

- `nav.goal.finish`：提交当前 Goal 的结果。复合任务可能返回下一个 Goal，Task 仍继续。
- `nav.stop`：Agent 宣布整个 Task 结束，触发 Harness 终止竞争。
- `Environment.wait_terminal()`：原生环境自行终止，例如动作预算耗尽或服务失败。
- `NavigationResult.terminal`：上述信号中第一个成功认领终止状态者。

因此 VLN 输出 STOP 不直接等于整个 Task 结束。VLN Job 先结束，Agent 再决定提交 Goal、进入下一
Goal、重试，或结束 Task。

## Profile、requirements 与 tool schema

这三个概念解决不同问题：

- `NavigationProfile` 描述 Environment 实际提供的观测通道、动作集合、单位和相机参数。
- `VLNNavigator.requirements` 描述模型正常运行必须满足的 profile。
- `Tool.input_schema` 校验某次调用的参数结构。

Profile 在组件启动阶段阻止不兼容组合；tool schema 在每次调用时阻止非法输入。二者不可互相替代。

## 验证等级

文档使用四类状态，不把“写了 adapter”视为已完成兼容：

| 等级 | 含义 |
|---|---|
| `contract` | 接口与 mock/单测成立 |
| `data_contract` | 官方 case 到原生配置/episode 的映射成立，但大型资源未完整运行 |
| real smoke / fixed trace | 真实依赖上跑过一个或固定若干 case，并保留证据 |
| official parity | 相同版本、split 和评分器与官方实现对照通过 |

当前三个真实 VLN 已达到 R2R-CE 固定三例 trace，官方 parity 仍是发布门禁。
