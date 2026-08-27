# Task 与 Goal 模型

导航 Harness 最重要的数据边界是 `BenchmarkCase` 与 `NavTask` 的分离。前者属于评测系统，后者
是 Agent 能看到的公开任务。

## 公共数据结构

```python
@dataclass(frozen=True, slots=True)
class NavGoal:
    goal_id: str
    instruction: str
    modality: str = "language"
    public: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class NavTask:
    task_id: str
    goal: NavGoal
    scene_id: str | None = None
    public: Mapping[str, Any] = field(default_factory=dict)
```

数据类不可变，插件不能在运行中悄悄改写题目。`modality` 表达语言、object、image 等目标类型；
模态特有的公开信息放入 `goal.public`，但稳定字段保持精简。

## 私有 Case

Bench loader 创建：

```python
BenchmarkCase(
    case_id="r2r:val_unseen:1",
    task=public_nav_task,
    setup={...},  # reset 环境所需的原生 episode、起点等
    truth={...},  # reference path、goal position、评分参数等
)
```

Runner 把整个 Case 交给 stack factory，使 Environment 能读取 `setup`；只把 `case.task` 传给
`NavigationHarness.run_task` 和 Agent。评分阶段才把 Case 与 `NavigationResult` 再交给 Bench。

::: danger 防止真值泄漏
不要为了构造 Environment 方便而把 `setup` 或 `truth` 合并到 `task.public`。新增 Bench 的契约
测试应断言 Agent 可见对象中没有起点之外的隐藏路径、目标坐标或未来目标。
:::

## Observation 明确时间与坐标系

```python
Observation(
    observation_id="42",
    source_time=...,
    received_time=...,
    frame="habitat_episode",
    channels={"rgb": ..., "depth": ..., "gps": ..., "compass": ...},
    pose=Pose(frame="habitat_episode", x=..., y=..., yaw=...),
    extras={...},
)
```

- `source_time` 是传感器/模拟器生成时间，`received_time` 是适配器接收时间。
- `frame` 必须显式，避免把 Habitat episode GPS、MP3D world 和 Isaac world 坐标直接混算。
- 大数组在跨进程 RPC 中由 file-backed media descriptor 传输，而不是膨胀 JSONL。
- `extras` 只承载非核心、可选的环境反馈；依赖它的插件应明确声明兼容条件。

## 复合任务状态机

GOAT 一类连续导航任务遵循：

```text
Task starts with Goal 0
  -> Agent/VLN navigates
  -> Agent calls nav.goal.finish(completed)
      -> done=false, return public Goal 1
  -> Agent continues in the same Environment session
  ...
  -> final goal.finish
      -> done=true
  -> Agent calls nav.stop(completed)
```

未来 Goal 由 Environment 私有保存，并且只有 `nav.goal.finish` 能逐个揭示。这样 Agent 的跨 Goal
记忆与物理位置连续，而又不会提前看到整个目标序列。

## Goal 失败并不自动结束 Task

`nav.goal.finish(status, reason)` 的具体接受语义由 Environment 决定；Agent 根据返回的
`accepted`、`done` 和可选 `goal` 决定下一步。框架不会假定所有 Bench 都允许跳过目标，也不会
把 VLN Job 的失败自动改成环境失败。基线 Agent 选择在 VLN Job 失败后调用 `nav.stop("failed")`，
自定义 Agent 可以实现重试或降级。

## 身份字段的粒度

- `case_id`：Run 内记录与排序键，由 Bench 保证可追踪。
- `task_id`：Agent 侧任务身份，可用于 Memory provenance。
- `goal_id`：复合 Task 内目标身份。
- `execution_id`：Harness 每次执行生成的 UUID，连接审计与结果。
- `observation_id`：Environment 内单调产生，不应假设跨 Task 全局唯一。
- `job_id`：VLN navigator 为一次委托生成。

不要用文件名或数组下标替代这些身份；多 shard 数据集可能存在重复原生 episode id。
