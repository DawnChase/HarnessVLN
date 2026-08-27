# 仓库地图

仓库不使用 `src/harnessvln/` 的额外安装层。运行时把根目录下的 `src/` 放入 `PYTHONPATH`，
一级目录直接表达模块职责。

```text
HarnessVLN/
├── src/
│   ├── harness/       # 运行时、ToolBus、配置、Runner、CLI
│   ├── schemas/       # 跨插件的最小导航数据类型
│   ├── agents/        # Agent Core 插件
│   ├── vln/           # 每个 VLN 独立子目录；共享 RPC 基础设施
│   ├── envs/          # 模拟器/真机中间件
│   ├── benches/       # 数据集 case loader 与评分
│   └── memory/        # 空间记忆插件
├── config/
│   ├── agents/ envs/ vln/ benches/ memory/ runs/
│   └── conda/harnessvln.yaml
├── scripts/           # 固定组合的可执行入口
├── tests/
│   ├── unit/          # 局部行为与竞态
│   ├── contract/      # 数据、模型配置、trace 契约
│   └── integration/   # 从 YAML 运行的闭环 smoke
├── docs/
│   ├── guide/         # 本站 Markdown 与 VitePress 源
│   ├── page/          # 生产静态站点
│   └── traces/        # 可审查的真实运行证据
├── data/              # 数据集、场景与资产；可含软链接
├── model/             # 模型 checkpoint
├── cache/             # 上游源码与可再生缓存，不提交
└── runs/              # Manifest、Memory 与运行产物，不提交
```

## 从入口向内阅读

推荐按调用关系而不是字母顺序阅读：

1. `harness/cli.py`：解析多个 YAML，决定进程退出码。
2. `harness/app.py`：构造 Bench、StackFactory、Runner，并写 Manifest。
3. `harness/runner.py`：有界调度完整 Task。
4. `harness/runtime.py`：单 Task 生命周期与终止竞争。
5. `harness/tool_bus.py`：能力注册、权限、校验、写屏障与审计。
6. `harness/contracts.py`：四类核心插件的最小 Protocol。

再根据改动方向进入 `agents/`、`vln/`、`envs/`、`benches/` 或 `memory/`。

## 为什么 VLN 必须有子目录

`src/vln/streamvln/`、`janusvln/`、`dualvln/` 各自拥有 navigator 与 worker。共享代码仅放在
`vln/rpc.py` 和 `vln/worker.py`。这样模型特有预处理、checkpoint 加载和内部循环不会堆到一个
条件分支文件，也不会被 Harness 强行抽象成同一内部结构。

## 配置不是代码注册表

YAML 通过 `module:object` factory 指向可调用对象，例如：

```yaml
stack:
  agent:
    factory: agents.passthrough:PassthroughVLNAgent
    params:
      poll_period_s: 1.0
```

增加插件通常无需修改中央枚举。稳定的是 Protocol、schema 与工具名字，具体实现由配置加载。

## 不应放进仓库的内容

`cache/`、模型/数据软链接、运行产物和根目录普通文件受 `.gitignore` 约束。可复现信息应进入配置
中的 `provenance`、紧凑 trace 或文档；大型 checkpoint、完整 manifest 和模拟器缓存保留在本地。
