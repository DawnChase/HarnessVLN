# 数据与控制流

架构是否低耦合，取决于跨边界流动的数据是否最少且可验证。本页从 Bench 输入一直跟踪到
Manifest 输出。

## 从数据集到 Agent

```text
dataset shard
  -> Benchmark.cases()
  -> BenchmarkCase(task, setup, truth)
       | task  -> NavigationHarness -> Agent
       | setup -> Environment factory/reset
       ` truth -> Benchmark.score (Task 结束后)
```

Runner 不把 Case 序列转换成统一大列表。loader 可以边解压边 yield；私有字段保持在 Case 内。

## 从 Environment 到模型

Environment 的 `nav.observe` 返回标准 envelope，`channels` 保留可扩展映射：

```json
{
  "observation_id": "12",
  "source_time": 1750000000.1,
  "received_time": 1750000000.1,
  "frame": "habitat_episode",
  "channels": {
    "rgb": "<ndarray or media descriptor>",
    "depth": "<ndarray or media descriptor>",
    "gps": [1.2, -0.4],
    "compass": [0.52]
  },
  "pose": {"frame": "habitat_episode", "x": 1.2, "y": -0.4, "yaw": 0.52},
  "extras": {}
}
```

VLN requirements 只列出真正依赖的 channel。JanusVLN 需要 RGB；StreamVLN 需要 RGB、Depth、
Pose 和 camera intrinsics；DualVLN 需要 RGB-D 与 -30° camera pitch。profile mismatch 在模型
启动前产生可读错误。

## 动作流

公开动作目前是 `nav.move.discrete(action)`。Environment 在自己的 profile 中声明可接受动作和
步长，再映射到原生系统：

| 公共动作 | Habitat | RoboTHOR | Isaac/InternUtopia |
|---|---|---|---|
| `forward` | action id / `move_forward` | `MoveAhead` | 高层 locomotion command，多 tick |
| `turn_left/right` | action id / task action | `RotateLeft/Right` | 高层 turn command，多 tick |
| `look_up/down` | profile 支持时映射 | `LookUp/Down` | 取决于 adapter |
| `stand_still` | Dual overlay 映射为无原生 stop | 不提供 | 可由 profile 扩展 |

工具 schema 的 enum 来自具体 Environment，而不是全局动作全集。不支持的动作在调用前即被拒绝。

## VLN RPC 数据流

重模型运行在独立 worker 时采用双向 JSONL 协议 v2：

1. navigator 通过 socketpair 启动 worker 并发送 `hello`；
2. navigator 发送 `navigate.start/status/cancel` 请求；
3. worker 发送 reverse `tool.call`；
4. navigator 经受限 ToolClient 调用 ToolBus，再回传结果；
5. numpy 图像/深度编码为临时 file-backed array descriptor；
6. Job 关闭后释放该 Job 的媒体文件。

协议与 stdout/stderr 分离，模型日志不会破坏 JSON 帧。worker 退出、超时或 malformed response 会
转换为带 stderr tail 的 RPC error。

## ToolEvent 审计流

每次调用尝试产生事件，包含 sequence、actor、name、摘要后的 arguments、开始时单调时间、outcome
与异常类别。事件进入 `NavigationResult.audit`，最终写到每个 Manifest record。审计用于：

- 重建观察/动作/停止顺序；
- 判断动作来自 Agent 还是 VLN；
- 识别 schema/权限失败和清理期间调用；
- 统计真实工具级耗时。

审计不是模型输入，也不包含 Benchmark 私有真值。

## 输出与可复现信息

Run 结束后 App 汇总指标并原子写 `manifest.json`：

```text
ResolvedConfig + source paths + SHA-256 digest + provenance
  + Bench name/split/validation status
  + ordered CaseRecord[]
      + terminal + environment result + metrics
      + runner error + cleanup errors + audit
  -> temporary file -> fsync -> os.replace -> manifest.json
```

即使进程读取 Manifest 的同时新结果写入，也只会看到旧完整文件或新完整文件，不会看到半截 JSON。

## 当前故意保留的开放字段

`NavGoal.public`、`NavTask.public`、`Observation.channels/extras` 和插件 `requirements` 使用映射，
用于容纳新模态和模拟器差异；身份、时间、坐标、工具名字和 lifecycle 则保持固定。扩展字段应：

1. 使用命名清晰、可 JSON 化的值；
2. 不复制已有稳定字段；
3. 在消费方声明 requirement；
4. 不承载评分真值；
5. 在 Manifest 或 provenance 中记录格式版本。
