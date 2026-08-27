# 结果与 Manifest

每次配置运行写出一个 `manifest.json`。它既是结果文件，也是复现与故障定位的主索引。

## 顶层结构

```json
{
  "schema_version": 1,
  "created_at_unix": 0.0,
  "config": {},
  "config_sources": [],
  "config_digest": "sha256...",
  "provenance": {},
  "benchmark": {
    "name": "r2r-ce",
    "split": "val_unseen",
    "validation_status": "..."
  },
  "aggregate_metrics": {},
  "records": []
}
```

`aggregate_metrics` 对所有 record 中存在的同名数值做算术平均。它不是加权聚合，也不会替代官方
evaluator 的整体定义；正式发布应由对应 Bench 明确验证聚合语义。

## 每个 record

| 字段 | 含义 |
|---|---|
| `index` / `case_id` | 输入顺序和稳定 case 身份 |
| `error` | stack 构造、Harness 或评分阶段的 runner-level 异常文本 |
| `metrics` | Bench 返回的逐 case 指标 |
| `execution_id` | Harness 执行身份；未进入 Harness 时为 null |
| `terminal` | Task 的 status、reason、actor |
| `environment` | trajectory、动作、native metrics 等适配器结果 |
| `cleanup_errors` | 主流程结束后资源清理失败；不覆盖 terminal |
| `audit` | 全部 ToolEvent |

## 三类失败必须分开看

```bash
jq '{
  task_failed: [.records[] | select(.terminal.status == "failed")] | length,
  runner_error: [.records[] | select(.error != null)] | length,
  cleanup_error: [.records[].cleanup_errors[]] | length
}' runs/EXPERIMENT/manifest.json
```

- **task failure**：任务确实进入 Harness，但 Agent、Environment 或 timeout 得到失败 terminal。
- **runner error**：该 case 没有可用 result，例如 factory 或 score 抛异常。
- **cleanup error**：已有主结果，但停止 worker、环境或 memory 时失败。

仅查看 SR 可能漏掉 cleanup error；CLI 因此把三类都纳入非零退出码。

## 查看控制权与动作来源

```bash
jq '.records[0].audit | sort_by(.sequence)[] |
  {sequence, actor, name, arguments, outcome, error}' \
  runs/EXPERIMENT/manifest.json
```

透传 Agent 的典型审计中，`vln.navigate.*` 与 `nav.goal.finish/stop` actor 是 agent，而
`nav.observe/nav.move.discrete` actor 是 vln。这直接证明 Runner 没有执行逐步控制。

## 核对配置身份

```bash
jq '{config_sources, config_digest, provenance, benchmark}' \
  runs/EXPERIMENT/manifest.json
```

digest 来自解析后配置的 canonical JSON，而不是源文件字节。因此等价 YAML 得到同一 digest；源文件
路径仍单独保留以追踪叠加顺序。

## 写入保证与覆盖行为

writer 在输出目录创建临时文件，写入、flush、`fsync` 后用 `os.replace` 原子替换
`manifest.json`。同一 output root 的后一次 Run 会覆盖主 Manifest；需要保留历史时给每次 Run
唯一目录，或在外层实验编排中归档。

## 紧凑 trace 与完整 Manifest

固定三例紧凑 trace 保存在本地 `project-notes/traces/`，包含动作摘要、terminal、最终 pose、
worker 复用与源 Manifest hash；该内部资料目录不随公开仓库提交。完整 Manifest 可能包含大量审计
和轨迹，留在 `runs/`。两者角色不同：trace 是受控证据，Manifest 是原始运行记录。

## 当前 schema 的兼容策略

`schema_version` 目前为 1。消费者应先检查版本，并容忍 record 中新增字段；移除或改变既有字段语义
时必须提升版本。不要依赖 JSON key 顺序。
