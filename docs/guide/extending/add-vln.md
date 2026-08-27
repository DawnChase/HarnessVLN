# 添加 VLN

新 VLN 应放在独立目录：

```text
src/vln/myvln/
  __init__.py       # 导出 navigator
  navigator.py      # Harness 侧 Job 与 requirements
  worker.py         # 可选：模型加载、内部循环、动作解析
```

不要把不同模型堆在共享 `vln/models.py` 的分支中。

## 1. 先写 requirements

从模型真实预处理和动作空间反推：

```python
class MyVLNNavigator(RPCVLNNavigator):
    model_name = "myvln"
    required_tools = frozenset({"nav.observe", "nav.move.discrete"})
    requirements = {
        "observation_channels": ["rgb", "depth"],
        "motion": {
            "tool": "nav.move.discrete",
            "actions": ["forward", "turn_left", "turn_right"],
            "forward_m": 0.25,
            "turn_deg": 15.0,
        },
        "camera": {"height": 480, "width": 640, "hfov_deg": 79},
    }
```

若模型要求特定 pitch、内参或 pose，明确写出。不要为了“组合成功”删除硬要求；应增加 Environment
overlay 或新 adapter。

## 2. 选择进程内或 RPC

| 方案 | 适用 | 参考 |
|---|---|---|
| 进程内异步 Job | 轻量策略、测试、无重依赖 | `vln/dummy/navigator.py` |
| RPC worker | CUDA 大模型、上游依赖、内部线程、多频推理 | 三个真实模型 |

进程内实现需要注册标准 start/status/cancel 三工具，并管理每个 Job 的 asyncio task。RPC 实现可继承
`RPCVLNNavigator`，worker 使用 `WorkerRuntime` 与 model-specific backend。

## 3. 保留完整模型循环

worker 的 navigate Job 可以这样组织：

```text
initialize model-owned history/state
while not cancelled and under budget:
    observation = reverse_call("nav.observe", {})
    update model buffers
    actions, stop = model.infer(...)
    for action in actions:
        reverse_call("nav.move.discrete", {"action": action})
    if stop: succeed job
```

若模型大小脑不同频，由 worker 内部调度线程/队列。外层 status 只报告 Job 状态，绝不要求每次
轮询只推进一步。

## 4. 归一化边界

model-specific worker 负责：

- checkpoint/tokenizer/vision tower 加载；
- Observation channel 到模型 tensor 的预处理；
- 历史帧和坐标变换；
- 模型文本输出到公共动作的 parser；
- 模型 STOP 到 Job succeeded 的转换。

Environment 负责原生动作映射。worker 不 import Habitat 或 Isaac，也不计算 Bench 指标。

## 5. 配置

```yaml
stack:
  vln:
    factory: vln.myvln:MyVLNNavigator
    scope: run
    params:
      command: [python, -m, vln.myvln.worker]
      checkpoint: model/myvln
      cwd: .
      env: {PYTHONPATH: src}
      worker_options:
        device: cuda:0
        max_steps: 500
      request_timeout_s: 900
provenance:
  model: MyVLN
  checkpoint_revision: COMMIT_OR_HASH
  upstream_commit: COMMIT
```

factory 参数必须与 navigator 构造函数一致。秘密、机器临时路径和下载 token 不写入提交配置。

## 6. run-scope 要求

只有 navigator 实现 `enable_run_scope()` 与 `close_run()` 才能配置 run scope。每个 Task 结束必须：

- 关闭/seal 所有 Job；
- 等待 reverse tool task 归零；
- 释放媒体文件；
- detach 旧 ToolClient；
- 清空会泄漏 case 信息的模型状态，保留只读 checkpoint。

模型是否保留跨 case KV/history 是算法协议问题；默认应清空，若研究跨 Task 模型记忆，必须显式配置
并写入 provenance，而不是借 run-scope 偶然泄漏。

## 7. 验收

先用 fake worker 测协议、超时、malformed response、取消和进程组回收，再做真实单帧推理、单 case
Harness、固定三例 worker 复用，最后做完整 split 和官方 parity。记录参数量或成功加载不能替代
导航闭环证据。
