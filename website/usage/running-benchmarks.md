# 运行 Bench

运行入口既可用 CLI，也可调用 Python API。两者最终走同一个配置解析、Runner、Harness 与
Manifest writer。

## CLI

```bash
export PYTHONPATH="$PWD/src"
python -m harness.cli CONFIG_1.yaml CONFIG_2.yaml ...
```

退出码规则：所有 case 的 task failure、runner error 和 cleanup error 都为零时返回 0；任一类非零
返回 1。配置错误、插件构造错误等 Run 级异常会直接以异常失败。

### Dummy 回归

```bash
bash scripts/run_dummy.sh
```

这是每次修改 Harness、ToolBus 或 Runner 后最先运行的端到端检查。

### 三个真实 R2R 模型

```bash
bash scripts/run_r2r_streamvln.sh
bash scripts/run_r2r_janusvln.sh
bash scripts/run_r2r_dualvln.sh
```

脚本默认面向完整 `val_unseen` 配置且超时较长。开发期应在命令末尾增加
`config/runs/smoke_one.yaml`，或创建单独 run override，不要改 Bench 基础配置。

## Python API

```python
import asyncio
from harness.app import run_config


async def main() -> None:
    summary, manifest = await run_config([
        "config/benches/dummy.yaml",
        "config/runs/dummy_passthrough.yaml",
    ])
    print(summary.benchmark, len(summary.records), manifest)


asyncio.run(main())
```

同步调用方可用 `run_config_sync(paths)`，但不能从已经运行的 event loop 内再次调用它。

## 批量执行的语义

- `parallelism` 控制完整 Task consumer 数；
- `max_cases` 在 case 流上截断，适合 smoke；
- 每个 case 的插件异常被记录为 `CaseRecord.error`，后续 case 仍可继续；
- 结果最终按输入序号排序，不按完成顺序；
- Run 结束无论成功失败都会尝试关闭 run-scoped VLN。

若 stack 要求串行而配置 `parallelism > 1`，Runner 在消费 case 前失败。不要把它当性能问题处理。

## 普通任务与连续任务

R2R/ObjectNav 的一个 case 通常只含一个 Goal。GOAT case 是一个含多个 Goal 的 Task：

```text
Runner starts one GOAT case once
Agent -> Goal 0 -> finish_goal -> Goal 1 -> ... -> final Goal -> stop
Environment session remains alive for the whole chain
```

批量并行仍发生在不同 GOAT compound task 之间，而不是同一 task 的 goals 之间。使用持久化 Memory
时当前配置要求串行，以保证跨 Task 写回顺序明确。

## 运行 Habitat / RoboTHOR / Isaac

这些入口必须组合匹配的 Bench 与 Environment 配置：

| Bench YAML | Environment YAML |
|---|---|
| `r2r_ce.yaml` | `habitat_r2r.yaml` |
| `goat.yaml` | `habitat_goat.yaml` |
| `habitat_objectnav_mp3d.yaml` | `habitat_objectnav_mp3d.yaml` |
| `habitat_objectnav_hm3d.yaml` | `habitat_objectnav_hm3d.yaml` |
| `robothor_objectnav.yaml` | `robothor.yaml` |
| `vln_pe.yaml` | `isaac_vln_pe.yaml` |
| `vlnverse.yaml` | `isaac_vlnverse.yaml` |

还需选择任务范式匹配的 Agent/VLN。当前三个真实 checkpoint 只验收了 R2R-CE；ObjectNav 与
GOAT 的环境/Bench smoke 不代表它们已有适用的策略插件。

RoboTHOR 固定旧 Unity build 在当前机器需 Xvfb + Mesa 软件 GLX。Isaac 两套配置均声明串行，且
目前缺专用大型资源，只进行 data-contract 开发。

## 如何停止长运行

正常中断会进入 asyncio 取消与组件清理路径。等待 CLI 返回，并检查 Manifest/错误输出中的
cleanup error。若外部强制杀死主进程，Harness 无法保证原子 Manifest 已写入；worker 进程组通常
随父进程清理，但仍应检查 GPU 和模拟器进程。
