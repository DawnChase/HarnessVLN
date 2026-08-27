# 添加空间记忆

Memory 插件应提供围绕导航空间的稳定能力，同时把存储、索引和融合算法藏在内部。

## 1. 定义导航记录

至少明确：

- `frame`：坐标系身份；
- `pose` 或空间锚点；
- `text/semantic`：可检索内容；
- `source_task_id`、时间与观测证据；
- confidence / version；
- 可见性范围与跨 scene 策略。

没有 frame 的位置不能安全进行近邻搜索。跨 scene 使用同名 `world` 也不够，应使用稳定 scene/map
命名空间或显式变换。

## 2. 实现最小 Protocol

```python
from harness.tool_bus import Tool, ToolClient


class TopologicalMemory:
    required_tools = frozenset({"nav.observe"})

    async def start(self, task, tools: ToolClient):
        self.task = task
        self.tools = tools
        await self.store.open()
        return (
            Tool("spatial.search", "Search navigation memory.", SEARCH_SCHEMA,
                 self._search),
            Tool("spatial.remember", "Store navigation evidence.", REMEMBER_SCHEMA,
                 self._remember, writes=True),
        )

    async def stop(self, reason: str) -> None:
        await self.store.flush_and_close()
```

若 Memory 需要读取当前 pose，可以调用受限 `nav.observe`；不要从 Environment 实例取私有字段。

## 3. 保持公共工具兼容

现有 `spatial.search(query, frame?, near_pose?, top_k)` 和
`spatial.remember(text, frame, pose?)` 适合最小 landmark 方法。复杂实现可以在返回 item 中增加
`score`、`node_id`、`evidence` 等字段，但不能改变现有字段语义。

若 Agent 确实需要原子图操作，可新增工具：

```text
spatial.neighbors(node_id, relation?, top_k)
spatial.route(source, target, constraints?)
```

新增前为两种 Memory 实现写相同 contract test，确认它是公共能力而不是存储引擎泄漏。

## 4. 跨 Task 持久化

Task-scope 实例不等于 Task-scope 数据。实例可连接同一个 SQLite/Postgres/vector service，start 时按
scene/task policy 创建 session，stop 时提交。

并行策略必须与存储保证一致：

| 存储 | 建议配置 |
|---|---|
| 单 JSON read-modify-write | `serial: true` |
| SQLite 单机事务 | 根据 journal/locking contract 测试后决定 |
| 集中服务/数据库 | 可并行，但需幂等 key 与事务 |
| 只读索引 | `writeback: false`，可并行 |

## 5. 与 Agent 的分工

Memory 返回可解释的空间证据，Agent 决定何时检索、如何放入 VLN 指令、何时写入和如何处理冲突。
自动压缩、反思或跨任务策略不应暗藏在基础 Memory 的 stop 中；若研究记忆演化，应作为显式 Agent
workflow 或独立策略层，使 Bench 能记录其触发条件。

## 6. 测试

- 空库、重复记录、frame filter、near-pose 排序、top_k；
- 非法/损坏持久化文件；
- stop 两次与 start 部分失败；
- 写入原子性和进程中断；
- 多 Task 可见性与 scene 隔离；
- 并发 writer 冲突；
- 不把 Bench truth 写入 Memory；
- schema 版本迁移和旧数据读取。

最后用 `SubtaskNavigationAgent` 或专用测试 Agent 跑两个连续 case，确认第二个 Task 能读取第一个
Task 写入的 landmark，并在 Manifest 中看到对应 tool actor 和顺序。
