# 配置叠加

CLI 接受一个或多个 YAML，并按命令行从左到右深度叠加。典型顺序是：

```text
Bench -> Agent -> Environment -> model-specific Env overlay -> VLN -> Run override
```

每个文件只描述一个关注点，因此换 Agent 不必复制环境参数，做单例 smoke 也不必修改正式 run。

## 一个完整组合

```bash
PYTHONPATH=src python -m harness.cli \
  config/benches/r2r_ce.yaml \
  config/agents/passthrough.yaml \
  config/envs/habitat_r2r.yaml \
  config/vln/streamvln.yaml \
  config/runs/r2r_streamvln.yaml \
  config/runs/smoke_one.yaml
```

最后的 `smoke_one.yaml` 把 `parallelism` 与 `max_cases` 覆盖为 1，其余字段保留。

## 顶层结构

```yaml
benchmark:
  factory: benches.r2r_ce:R2RCEBenchmark
  params: {root: data/datasets/r2r, split: val_unseen}

stack:
  agent: {factory: agents.passthrough:PassthroughVLNAgent, params: {}}
  environment: {factory: envs.habitat:from_case, params: {}}
  vln: {factory: vln.streamvln:StreamVLNNavigator, scope: run, params: {}}
  memory: null

runner:
  parallelism: 1
  max_cases: 1
  timeout_s: 3600
  shutdown_timeout_s: 30

output:
  root: runs/example

provenance:
  combination: r2r-stream-example
```

配置加载后由 JSON Schema Draft 2020-12 校验，随后生成 canonical JSON 的 SHA-256 digest。
解析后的完整配置、源文件绝对路径和 digest 都写入 Manifest。

## ComponentSpec

每个组件支持：

| 字段 | 含义 | 默认值 |
|---|---|---|
| `factory` | `module:object` 可调用对象 | 必填 |
| `params` | 构造函数关键字参数 | `{}` |
| `scope` | `task` 或 VLN 可用的 `run` | `task` |
| `serial` | 声明进程级/共享资源要求串行 | `false` |

Environment factory 额外收到 `case=BenchmarkCase`。其他 `params` 原样传给 factory，不需要在核心
代码增加插件专用字段。

## 深度叠加与 factory reset

普通映射递归合并：

```yaml
# base
stack:
  environment:
    factory: envs.habitat:from_case
    params:
      forward_m: 0.25
      camera: {width: 640, height: 480, pitch_deg: 0}
```

```yaml
# overlay
stack:
  environment:
    params:
      camera: {pitch_deg: -30}
```

结果只覆盖 pitch。**如果同一映射中的 factory 被换成另一个值，该组件子树从空映射重新叠加**，
避免旧实现的 params 泄漏给新 factory：

```yaml
stack:
  environment:
    factory: envs.ai2thor:RoboTHOREnvironment
    params: {...}  # 必须重新给出需要的参数
```

列表不合并，后值整体替换前值。

## DualVLN 的专用 overlay

DualVLN 要求 -30° RGB-D 和 `stand_still`，因此配置顺序不能省略：

```text
habitat_r2r.yaml
  -> habitat_r2r_dualvln.yaml  # 调整 sensor orientation、动作、profile
  -> dualvln.yaml              # 声明对应 requirements 与 worker
```

这是 profile 驱动兼容的示例：模型差异没有写进共享 Habitat adapter 的条件分支。

## scope 与 parallelism

配置加载会拒绝 run-scoped Agent、Environment、Memory。VLN 可使用 `scope: run`，但当前
StackFactory 将其视为 requires-serial。任一组件 `serial: true` 或 Memory writeback 启用时，
`runner.parallelism` 必须为 1。

## seed 的当前边界

`runner.seed` 当前会进入解析配置、digest 和 Manifest，但 runtime 尚未统一注入 Python、NumPy、
Torch、Environment 与 worker。因此它是实验意图记录，不是全栈确定性保证。需要复现实验时，还应
在具体插件/模拟器配置中固定 seed，并记录上游版本。后续若实现统一 seed service，必须保留插件
显式覆盖能力。

## provenance 应记录什么

建议记录 model/checkpoint/upstream commit、dataset revision/hash、simulator build、adapter version、
组合名与验证范围。`provenance` 不参与运行逻辑，但进入 Manifest，不能用它替代真实配置字段。
