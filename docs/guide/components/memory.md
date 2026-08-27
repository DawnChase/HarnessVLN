# 空间记忆

Memory 在 HarnessVLN 中不是通用聊天记录，而是围绕导航位置组织的可查询知识。基础版本提供
`DummyLandmarkMemory`，用于验证工具、持久化和跨 Task 行为，不声称实现完整地图算法。

## 契约

```python
class SpatialMemory(Protocol):
    required_tools: frozenset[str]

    async def start(self, task: NavTask, tools: ToolClient) -> Sequence[Tool]: ...
    async def stop(self, reason: str) -> None: ...
```

Memory 可以通过自己的受限 ToolClient 读取环境能力，但基础实现无需额外工具。它向 Agent 注册
`spatial.search` 和 `spatial.remember`。

## Dummy Landmark 数据模型

每条记录保存：

```json
{
  "id": "...",
  "text": "turn at the red sofa",
  "frame": "habitat_episode",
  "pose": [1.2, -0.4, 0.52],
  "source_task_id": "r2r:val_unseen:1"
}
```

`pose` 可缺省。`frame` 必填，阻止跨坐标系直接比较距离；`task_id` 提供来源而不限制跨 Task 查询。

## 查询语义

```python
await context.spatial.search(
    query="red sofa",
    frame="habitat_episode",
    near_pose=[1.0, -0.5, 0.4],
    top_k=5,
)
```

基础实现按文本匹配和可选欧氏近邻排序，并支持 frame 过滤。它不生成 embedding、不做坐标变换、
不融合重复 landmark，因而只适合作为占位和接口测试。

## 持久化与跨 Task

配置：

```yaml
stack:
  memory:
    factory: memory.dummy_landmark:DummyLandmarkMemory
    serial: true
    params:
      root: runs/memory/dummy_landmark
      writeback: true
```

每个 Task 创建一个 Memory 实例，start 时读取 `landmarks.json`，stop 时通过临时文件、`fsync` 和
`os.replace` 原子写回。知识因此跨 Task 和 Run 保存，但连接与 Task 上下文不复用。

单文件 read-modify-write 不支持并发 writer，所以 `writeback: true` 使 stack 要求串行。只读副本可
关闭 writeback；真正并发需要数据库事务或集中 memory service。

## 与复合任务的关系

GOAT 的多个 Goal 在一个 Task 中共享同一个 Memory 实例，因此不需要落盘就可在目标间查询。
Task 结束后写回，后续 case 才能看到。若研究严格的 online lifelong protocol，应在 Bench 中定义
何时允许读写、是否按 scene 隔离，并把该策略放进 provenance。

## 演进方向

新 Memory 可以实现：

- 拓扑节点与可通行边；
- metric occupancy / semantic map；
- landmark embedding 检索；
- 多层 episodic 与 consolidated memory；
- 跨坐标系锚点与地图对齐。

它们仍应围绕 `frame + pose + navigation evidence` 设计。模型摘要、提示词压缩或泛化对话存储不应
伪装为空间记忆；若未来需要，应作为独立 Agent state/evolution 插件讨论。

## 扩展接口的原则

当前 `search/remember` 是最小稳定面。复杂方法可先把图结构或置信度放进 item 扩展字段；只有多个
Agent 确实需要同一种原子操作时，再新增如 `spatial.neighbors` 或 `spatial.route`。Memory 内部
存储格式可以变化，公共工具结果必须有版本和 contract test。
