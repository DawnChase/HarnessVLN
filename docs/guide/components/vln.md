# VLN 插件

VLN 插件把一条导航指令变成一个可管理的异步 Job。Harness 不拆模型内部循环，也不要求一次推理
对应一个动作。

## 最小契约

```python
class VLNNavigator(Protocol):
    required_tools: frozenset[str]
    requirements: dict[str, Any]

    async def start(self, task: NavTask, tools: ToolClient) -> Sequence[Tool]: ...
    async def stop(self, reason: str) -> None: ...
```

`start` 得到的 ToolClient 只能访问 `required_tools`。navigator 返回供 Agent 调用的三个工具：
`vln.navigate.start`、`vln.navigate.status`、`vln.navigate.cancel`。

## Job 状态机

```text
start -> running -> succeeded
                 -> failed
                 -> cancelled
```

`navigate.start` 立即返回 `job_id`，耗时推理在后台进行。`status` 返回稳定状态和 reason；`cancel`
必须可重复调用。Job 成功表示模型认为该段指令结束，是否提交 Goal 由 Agent 决定。

## 三个当前模型

| 模型 | 输入 requirement | 动作 | 特殊 profile |
|---|---|---|---|
| StreamVLN | RGB、Depth、Pose、camera intrinsics | forward / left / right | 640×480，HFOV 79°，15° turn |
| JanusVLN | RGB | forward / left / right | 640×480，HFOV 79°，15° turn |
| DualVLN | RGB、Depth | stand_still / forward / left / right | camera pitch -30° |

每个模型位于 `src/vln/<model>/`，拥有独立 navigator 和 worker。配置固定 upstream、checkpoint
revision、worker command、设备和推理选项。

## 为什么使用独立 worker

当前三个真实模型通过 `RPCVLNNavigator` 启动独立进程：

- 隔离重型 import、CUDA 生命周期和上游代码；
- stdout/stderr 与协议通道分离；
- 可以终止整个进程组，避免模型子进程残留；
- run-scope 时只加载一次 checkpoint；
- Harness 主进程仍保持统一 Python API。

RPC 使用继承的 socketpair 文件描述符传双向 JSONL。`hello` 协商协议 v2，worker 收到 Job 请求，
也能反向发出 `nav.observe` / `nav.move.discrete` 调用。

## 大数组传输

RGB-D ndarray 不直接 JSON 编码。navigator 的 `FileArrayStore` 把数组写入临时文件，将 dtype、
shape、path 等 descriptor 发给 worker。文件按 Job 归属；`close_job` 等待反向调用结束后释放。
worker 不应长期持有已完成 Job 的 descriptor。

## run-scope 复用

真实模型 YAML 使用：

```yaml
stack:
  vln:
    factory: vln.streamvln:StreamVLNNavigator
    scope: run
```

第一次 case 创建并启动 worker；随后 case 重用 navigator 与 checkpoint，但每个 Task 重新绑定受限
ToolClient。上一个 Job 必须 sealed、反向调用归零且 media 释放后才能解绑。Run 结束统一
`close_run()`。

当前 run-scope 强制 `parallelism: 1`。并行大模型评测应创建多个 worker slot 的显式资源池，不能
让一个单连接 worker 同时绑定两个 Environment。

## requirement 校验

Environment 启动后，Harness 比较其 `NavigationProfile` 与模型 requirements：

- 必需 observation channels 是否存在；
- motion tool 与 action 集合是否满足；
- `forward_m`、`turn_deg` 等是否匹配；
- camera height、width、HFOV、pitch 等是否匹配。

例如 DualVLN 若缺少 `habitat_r2r_dualvln.yaml`，会在加载模型前因 -30° pitch 或
`stand_still` 缺失失败。这比模型运行数步后才发现输入分布错误更可靠。

## 模型内部仍归模型所有

以下内容不进入公共 VLN Protocol：prompt template、tokenizer、帧采样、动作 parser、KV cache、
System 1/System 2 调度、连续轨迹展开、停止 token。每个 worker 把这些细节收拢在自己的子目录，
只把 Job 状态和导航工具调用暴露给 Harness。
