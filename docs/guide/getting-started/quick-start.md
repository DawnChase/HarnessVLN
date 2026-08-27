# 十分钟运行

先运行不依赖 GPU、数据集和模拟器的 dummy 链路。它使用与真实任务完全相同的 Runner、Harness、
Agent、VLN Job、ToolBus、评分和 Manifest 路径，适合验证框架而不是验证模型性能。

## 1. 准备 Python 环境

完整环境统一叫 `harnessvln`，首发锁定 Python 3.10：

```bash
conda env create -f config/conda/harnessvln.yaml
conda activate harnessvln
export PYTHONPATH="$PWD/src"
```

只运行 dummy 和单元测试时，可在已有 Python 3.10+ 环境安装 `PyYAML`、`jsonschema` 和
`pytest`，无需安装模拟器与模型依赖。

## 2. 运行零依赖闭环

```bash
bash scripts/run_dummy.sh
```

也可以直接看到 YAML 的组合顺序：

```bash
PYTHONPATH=src python -m harness.cli \
  config/benches/dummy.yaml \
  config/runs/dummy_passthrough.yaml
```

成功时终端会报告 case 数与三类失败计数，并打印 Manifest 的绝对路径：

```text
dummy/smoke: 2 cases, 0 task failures, 0 runner errors, 0 cleanup errors
/home/csl/Project/HarnessVLN/runs/dummy_passthrough/manifest.json
```

检查关键结果：

```bash
jq '{benchmark, aggregate_metrics, records: [.records[] | {
  case_id, terminal, metrics, cleanup_errors
}]}' runs/dummy_passthrough/manifest.json
```

## 3. 理解这次运行发生了什么

1. `DummyBenchmark` 产生两个 `BenchmarkCase`。
2. Runner 最多并发两个完整 case。
3. 每个 case 创建独立 Agent、Environment 和 VLN 实例。
4. `PassthroughVLNAgent` 启动一个 VLN Job并轮询状态。
5. Dummy VLN 通过反向工具调用主动观察和移动。
6. Agent 完成当前 Goal，再调用 `nav.stop` 结束 Task。
7. Bench 使用私有 target 评分；Manifest 保留配置、终止原因、指标和工具审计。

Runner 在第 4 至第 6 步没有参与 observe-act。

## 4. 运行一个真实组合

真实 R2R 脚本已固定配置顺序、Conda 环境与输出目录：

```bash
bash scripts/run_r2r_streamvln.sh
bash scripts/run_r2r_janusvln.sh
bash scripts/run_r2r_dualvln.sh
```

先做单例 smoke 时，把 `config/runs/smoke_one.yaml` 放在命令最后：

```bash
PYTHONPATH=src python -m harness.cli \
  config/benches/r2r_ce.yaml \
  config/agents/passthrough.yaml \
  config/envs/habitat_r2r.yaml \
  config/vln/streamvln.yaml \
  config/runs/r2r_streamvln.yaml \
  config/runs/smoke_one.yaml
```

DualVLN 还需要在基础 Habitat 环境后叠加专用传感器 profile：

```text
... habitat_r2r.yaml -> habitat_r2r_dualvln.yaml -> dualvln.yaml -> run.yaml
```

::: warning 运行前置条件
真实脚本要求对应数据、场景、checkpoint、上游源码和 GPU 环境已经按配置路径落地。脚本成功启动
不等于完整 split 已通过官方 evaluator 对照；验证等级见[兼容与验证矩阵](../reference/compatibility.md)。
:::

## 5. 运行回归测试

```bash
conda run -n harnessvln pytest -q
```

当前基线为 147 项测试。新增插件至少应运行相关 unit、contract 和 configured smoke；改动共享
生命周期或 ToolBus 时运行全量测试。

## 6. 构建本指南

文档工具链与 Python 运行环境独立：

```bash
cd docs/guide
npm install
npm run docs:build       # 输出到仓库根目录 page/
npm run docs:dev         # 本地热更新
```

下一步：[术语与边界](./concepts.md)或[总体分层](../architecture/overview.md)。
