# 故障定位

先确定失败在哪一层，再读取对应证据。不要用“模型没走对”概括配置、环境、RPC 和评分问题。

## 快速分流

| 现象 | 首先检查 | 常见原因 |
|---|---|---|
| 配置加载前失败 | 异常中的 YAML 路径与字段位置 | 缺顶层字段、factory 语法、scope 非法 |
| `requires unavailable tools` | required_tools 与 Environment/Memory/VLN bindings | 组合缺插件、工具名拼错 |
| `RequirementMismatch` | Environment profile 与 VLN requirements | channel、动作、步长、相机 pitch 不匹配 |
| record `error` 非空 | Manifest record.error | factory、run_task 或 score 抛异常 |
| terminal failed | terminal.actor/reason + audit | Agent 异常、环境 terminal、Task timeout |
| 指标正常但 CLI 返回 1 | cleanup_errors | worker/env/memory 清理失败 |
| worker request timed out | stderr tail、request method、GPU | 模型加载/推理卡住、协议无响应 |
| 第二个 case 失败 | scope、旧 Job/media/ToolClient | run-scope 未释放任务状态 |
| `this stack requires serial` | serial/scope/memory writeback | parallelism 与资源契约冲突 |

## 配置与 factory

查看最终来源和 digest：

```bash
jq '{config_sources, config_digest, config}' runs/EXPERIMENT/manifest.json
```

若尚未写 Manifest，按原命令顺序逐个检查 YAML。factory 必须恰有一个冒号并指向可调用对象。
切换 factory 的 overlay 会重置整个组件子树，因此新 params 必须完整给出。

## Profile mismatch

错误会列出模型要求和环境实际值。典型修复方向：

- StreamVLN 缺 pose/camera_intrinsics：使用完整 R2R Environment 配置；
- DualVLN 缺 stand_still 或 pitch -30°：增加专用 env overlay；
- turn 30° 对模型要求 15°：调整真实 simulator profile，不能只改声明；
- RGB shape/HFOV 不符：同时调整 sensor config 和 profile camera。

不要通过删除 requirements 让启动通过。

## 分析 Manifest

```bash
jq '.records[] | {
  case_id, error, terminal, metrics, cleanup_errors,
  last_events: (.audit[-8:] // [])
}' runs/EXPERIMENT/manifest.json
```

最后事件能说明停止前谁在调用什么。`outcome` 为 denied/invalid/closed/error 时，结合 name、actor、
arguments 与 error 判断权限、schema、写屏障或 handler 问题。

## Agent 返回但任务失败

若 reason 表示 Agent returned without calling stop，Agent 的某条正常分支直接 return。所有业务终点
都应调用 `nav.stop`；外部取消分支可直接返回，因为 Harness 已进入终止路径。

## VLN worker

RPC 错误通常附最近 stderr。依次检查：

1. command/cwd/PYTHONPATH 是否与配置一致；
2. upstream root 与 checkpoint 是否存在；
3. worker `hello` 的 protocol/model/capabilities；
4. GPU OOM、dtype 和 meta tensor；
5. Observation channel/shape；
6. 动作 parser 输出是否在 Environment enum；
7. request timeout 是否覆盖真实加载时间。

只调大 timeout 会掩盖死锁。先确认 stderr 和 GPU 利用率仍有进展。

## Habitat

- scene not found：核对 `data/scene_datasets` 软链接和 scene id rewrite；
- sensor missing：检查 Hydra `config_options/config_values` 是否成功应用；
- 距离/坐标异常：不要混用 episode GPS 与 MP3D world；
- Dual 输入异常：核对 sensor orientation 和 profile pitch 同时为 -30°。

## RoboTHOR

旧 Unity build 在当前硬件需要独立 Xvfb 与 Mesa：

```bash
Xvfb :44 -screen 0 1024x768x24 -ac -noreset +extension GLX +render +iglx
DISPLAY=:44 LIBGL_ALWAYS_SOFTWARE=1 YOUR_COMMAND
```

`lastActionSuccess=false` 的受阻 Move 是正常反馈，是否重规划属于 Agent，不应自动归为 runtime error。

## Isaac / InternUtopia

- 当前两套专用 Bench 若报 scene/USD/policy 缺失，状态与矩阵一致；
- SimulationApp 是进程级资源，保持 `parallelism: 1`；
- 高层动作超过 tick 上限说明 locomotion completion signal 未出现，检查 native observation mapping；
- Kit 首次启动扩展解析较慢，区分首次缓存与持续无响应。

## 清理残留

cleanup error 出现后先保存 Manifest，再检查 worker、Unity/Kit 和 GPU 进程。根因修复应使插件 stop
幂等且有界；不要在 Runner 外简单吞掉清理异常，否则后续 case 可能绑定到污染资源。

## seed 与复现

当前 `runner.seed` 尚未统一注入各后端。指标不一致时同时核对 simulator、dataset、checkpoint、
worker options、Agent、成功阈值和各插件 seed。Manifest digest 相同只证明 Harness 配置相同，
不证明所有外部随机源相同。
