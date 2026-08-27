# 测试与发布门禁

测试按风险分层。纯 mock 单测保证生命周期不变量，真实 smoke 保证依赖能工作，官方 parity 保证
结果语义一致；任何一层都不能替代另一层。

## 当前测试布局

```text
tests/
  unit/         # runtime、ToolBus、Runner、RPC、各 adapter/worker
  contract/     # dataset loader、模型配置、trace artifact
  integration/  # 多 YAML 到 Manifest 的完整 dummy 链路
  fixtures/     # RPC/SDK fake worker
```

当前全量基线为 147 项：

```bash
conda run -n harnessvln pytest -q
```

## 按改动范围运行

```bash
# Agent / 基础导航插件
pytest -q tests/unit/test_navigation_plugins.py

# ToolBus 与单 Task 生命周期
pytest -q tests/unit/test_core_runtime.py

# Runner、配置与 run-scope
pytest -q tests/unit/test_runner.py tests/unit/test_app.py \
  tests/unit/test_config_requirements.py

# VLN transport 与 worker
pytest -q tests/unit/test_vln_rpc.py tests/unit/test_vln_worker.py \
  tests/unit/test_streamvln_worker.py tests/unit/test_janusvln_worker.py \
  tests/unit/test_dualvln_worker.py

# 数据与真实 trace 的静态契约
pytest -q tests/contract

# YAML 全闭环
pytest -q tests/integration/test_configured_smoke.py
```

修改共享 runtime、ToolBus、RPC 或 schema 后必须跑全量。

## 生命周期不变量

单测重点覆盖正常路径之外的竞态：

- Agent 正常返回但未 stop；
- Environment terminal 与 Agent stop 同时到达；
- 外部取消传播且 Environment 先停止；
- stop ACK 后在途 move 被 native generation fence 拒绝；
- 清理超时/异常不覆盖主 terminal；
- run-scoped worker 只启动/关闭一次；
- RPC spawn、hello、request、reaper 任意阶段取消；
- worker 子孙进程被进程组 fence 回收；
- Job media 在每个 case 后释放；
- status/cancel 并发只 finalize 一次。

新增异步插件至少复用这些故障模型设计自己的测试。

## 数据 contract

loader 测试应验证文件选择、数量/去重、稳定 ID、公开/私有字段和损坏输入。环境 adapter 测试使用
fake native session 验证动作映射、坐标、Observation envelope、goal transition 和 result。

资源 revision 与 config requirement 测试防止 YAML 被无意改成“能加载但输入分布不同”的组合。

## 真实验证阶梯

1. **依赖导入**：固定 Python/CUDA/库版本可 import。
2. **资源加载**：checkpoint 全部分片、无残留 meta tensor。
3. **单帧推理**：真实预处理与动作 parser。
4. **模拟器 smoke**：真实 reset/render/action/stop。
5. **Harness case**：Agent -> Job -> reverse tools -> Goal -> stop -> Manifest。
6. **run-scope 小样本**：固定三例、同 PID、逐 Job 清理。
7. **完整 split parity**：同官方版本、seed、阈值和 evaluator。

每一级保留命令、配置 digest、revision 与结果。只通过第 2 级不能声明模型已接入导航闭环。

## 固定 trace 门禁

本地 `project-notes/traces/*run-scope-3.json` 由可选 contract test 检查 schema、case 数、指标聚合、
动作摘要、cleanup、worker 复用和源 Manifest hash。公开仓库不包含这些内部运行资料；目录存在时
测试自动启用。更新 trace 必须来自新的可验证运行，不手工调整指标。

## 文档门禁

```bash
cd website
npm run docs:build
```

VitePress 构建会检查内部链接。生产页面输出到仓库根目录 `docs/`；提交文档改动时同时提交 Markdown 源、
生成图与静态构建结果，并在桌面和移动 viewport 检查导航、表格、代码块和长文本。

## 发布清单

- 全量 pytest 通过；
- dummy CLI 返回 0，Manifest 三类失败均为 0；
- 目标真实组合达到声明的验证等级；
- 无残留 worker/simulator/GPU 进程；
- 配置 `provenance` 与资源 revision 完整；
- 新公共 schema 有向后兼容说明；
- 文档矩阵没有把 pending 写成 complete。
