# 统一运行环境

项目只使用一个 Conda 环境 `harnessvln`，要求 Python >= 3.10；首发环境因模拟器兼容性
固定为 Python 3.10。Agent、VLN worker、
Bench 和环境中间件即使采用独立进程，也必须从该环境启动；进程隔离只用于生命周期和
GPU 资源隔离，不再用多套 Python 依赖规避接口问题。

基础环境由 `config/conda/harnessvln.yaml` 创建。Blackwell GPU 栈需在基础依赖完成后安装，
以保证 FlashAttention 构建阶段能够发现 Torch：

```bash
conda env create -f config/conda/harnessvln.yaml
conda activate harnessvln
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchvision==0.23.0
MAX_JOBS=8 python -m pip install flash-attn==2.8.3 --no-build-isolation
```

统一版本选择以真实模型加载为准。三个 VLN 共用 `transformers==4.51.0`；DualVLN 使用
`diffusers==0.32.2`，因为其 checkpoint 的 NextDiT FFN 权重为 `384x1024`，而 0.33.1
会构造不兼容的 `384x1536` 层。当前已验证 Python 3.10.20、Torch 2.8.0+cu128、
FlashAttention 2.8.3，且 DualVLN 四个权重分片加载完成、残留 meta 参数为 0。

DualVLN 的完整 Harness 链路已在 R2R-CE `val_unseen` episode 1 验证。专用环境 overlay
提供 -30° RGB-D 与不触发原生 STOP 的 `stand_still`；模型保留内部 System 1/System 2
异步循环，共执行 54 次 observe、53 个逻辑动作（含 7 次 stand-still）后停止。结果为
SR 1、SPL 0.936、NE 0.067、OS 1，且无清理错误；紧凑 trace 见
`docs/traces/dualvln-r2r-val-unseen-1.json`，完整 manifest 位于
`runs/r2r_dualvln/manifest.json`。该 smoke 不代表 VLN-PE/VLNVerse 已完成真实 episode 验收。

run-scope 另以连续两个 `val_unseen` episode 验证：环境按 Task 重建，DualVLN 从首个 Task
到第二个 Task 始终为同一 worker PID `2048182`，checkpoint 只加载一次；两个 Task 分别
执行 53/62 个动作，均 SR 1、OS 1，且均无 cleanup error。Bench 结束后 worker 和逐 job
RGB-D 临时文件均已清理。紧凑 trace 见
`docs/traces/dualvln-r2r-val-unseen-run-scope-2.json`，完整 manifest 位于
`runs/r2r_dualvln_run_scope_two/manifest.json`。

StreamVLN 已用同一环境完成本地离线加载与单帧 RGB-D 推理：主 checkpoint revision 为
`f1f76c66083c362ddfcd2610167f9c4e4a46c027`，四个 safetensors 分片覆盖索引中的
764 个 tensor；SigLIP revision 为 `9fdffc58afc957d1a03a25b10dba0329ab15c2a3`。
实际构造得到 8,030,345,248 个参数、残留 meta 参数为 0，测试输入解析出动作
`(turn_right, turn_right, turn_right, turn_right)`。运行参数见
`config/vln/streamvln.yaml`。

完整 Harness 链路另在 R2R-CE `val_unseen` episode 1 验证：Passthrough Agent 启动一次
VLN job，StreamVLN 主动执行 45 次 observe 和 44 个离散动作后输出 STOP，再由 Agent 调用
`nav.goal.finish` 与 `nav.stop`。结果为 SR 1、SPL 1、NE 0.581、OS 1，且无清理错误；紧凑
trace 见 `docs/traces/streamvln-r2r-val-unseen-1.json`，完整 manifest 位于
`runs/r2r_streamvln/manifest.json`。该 smoke 不替代固定三例对照或完整 split 验收。

JanusVLN Base 也已在该环境离线加载并完成单帧 RGB 推理。ModelScope checkpoint revision
固定为 `33f932a4ea6bdc34afca9f5b79a8b4537cd02509`；四个分片的 SHA-256 与官方清单一致，
2068 个 tensor 完整覆盖索引。实际模型包含 9,314,832,390 个参数和 24 层 VGGT，残留
meta 参数为 0，测试输入生成 `TURN_RIGHT`。为保持官方 Transformers 4.50 的图像处理
行为，统一环境中的 adapter 显式设置 `use_fast=False`；运行参数见
`config/vln/janusvln.yaml`。

完整 Harness 链路也在同一 R2R-CE episode 1 验证：JanusVLN 主动执行 43 次 observe 和
42 个动作后输出 STOP，Agent 再完成 goal 与 task。结果为 SR 1、SPL 1、NE 1.460、OS 1，
无清理错误；紧凑 trace 见 `docs/traces/janusvln-r2r-val-unseen-1.json`，完整 manifest 位于
`runs/r2r_janusvln/manifest.json`。该 smoke 同样不替代固定三例对照或完整 split 验收。

Habitat-Sim/Lab 固定为 v0.3.3（Sim commit
`acbe6f4922e68145e401e55c30f9dfea460a3f24`，Lab commit
`094d6be2f9d057e4781a68ae792132895fd4d3d0`）。Sim 以 headless、无 Bullet 模式从源码编译；
Lab/Baselines 从固定提交源码加载，避免它的旧 `gym<0.23.1` 包元数据破坏统一环境。Harness
在 adapter 边界兼容 Lab 0.3.3 使用的旧 registry 访问，环境仍保留 `gym==0.26.2`：

```bash
git clone --recursive --branch v0.3.3 https://github.com/facebookresearch/habitat-sim \
  cache/upstream/habitat_sim
git clone --branch v0.3.3 https://github.com/facebookresearch/habitat-lab \
  cache/upstream/habitat_lab
HEADLESS=True WITH_BULLET=False CMAKE_BUILD_PARALLEL_LEVEL=32 \
  python -m pip install cache/upstream/habitat_sim --no-build-isolation
```

已通过 `config/benches/r2r_ce.yaml + config/envs/habitat_r2r.yaml` 在 MP3D R2R
`val_unseen` episode 1 验证真实 reset、RGB-D render、navmesh、前进、转向与 stop。观测包含
`640x480` RGB-D、GPS、Compass、标准 pose 和相机内参；前进一步为 0.25 m，左转为 15°。
adapter 以 episode 内最小 geodesic distance 和 3 m 阈值补齐官方 OS 定义。该结果只确认
Habitat/R2R 环境链路；完整 split 评分、GOAT 与三个模型的 episode 对照仍按 release gate
分别验收。

Habitat ObjectNav 也使用同一 adapter，但保留独立 Bench 与 YAML。MP3D v1 和 HM3D v2
validation 首个 shard episode 均已完成真实 reset、`640x480` RGB-D、ObjectGoal、GPS、
Compass、前进、转向、stop 与 native SR/SPL smoke。HM3D 数据中的 `hm3d_v0.2` scene 前缀
和 scene-dataset config 只在 native session 边界映射到现有 HM3D 资源；Bench case id 使用
官方 loader 的 shard 内重编号，原始且可能重复的 episode id 只保存在私有 setup。

GOAT-Bench 固定源码 commit `74c41d19d4a4c3608d1575b512087b5a529aee0e`，仅在 adapter
边界加载官方 dataset、task 与 measurement 模块。`val_unseen` episode
`goat:val_unseen:4ok3usBNeis:0` 已以单个 Habitat session 执行转向、前进和连续两次
`subtask_stop`，取得逐目标 distance、success、SPL；10 个未来目标由环境私有持有并逐个
揭示。image-first episode `4ok3usBNeis:3` 的目标图像真实渲染为 `640x360 uint8`，重复观测
命中缓存，且渲染前后 agent 位姿不变。配置见 `config/benches/goat.yaml` 和
`config/envs/habitat_goat.yaml`。

AI2-THOR 固定为 2.7.2，RoboTHOR 2021 官方 Unity build 固定为
`bad5bc2b250615cb766ffb45d455c211329af17e`。统一环境同时固定
`opencv-python==4.11.0.86`，避免 pip 选择要求 NumPy 2 的新版 OpenCV。该旧 build 在当前
硬件 Xorg 上创建 GLX context 会返回 `BadAlloc`，已用独立 Xvfb 和 Mesa 软件 GLX 完成
真实 `640x480` RGB、Depth、Teleport、Rotate、Move、Stop 验证：

```bash
Xvfb :44 -screen 0 1024x768x24 -ac -noreset +extension GLX +render +iglx
DISPLAY=:44 LIBGL_ALWAYS_SOFTWARE=1 python -m harness.cli <BENCH_YAML> <RUN_YAML>
```

验证 episode 为 `FloorPlan_Train1_1_AlarmClock_0`；Harness 审计顺序为
`observe, turn, move, observe, goal.finish, stop`，blocked Move 被保留为动作反馈而没有错误
终止任务。配置片段见 `config/benches/robothor_objectnav.yaml` 和
`config/envs/robothor.yaml`。这确认 Environment/Bench 生命周期，不代表已有三个 R2R
checkpoint 能直接解决 ObjectNav，也不替代完整 validation split 的官方评分对照。

Isaac Sim 固定为 4.5.0，使用 NVIDIA pip index 安装在同一个 `harnessvln` 环境。首次启动
完成 Kit extension 的解析与缓存；随后用原生 `SimulationApp` 创建 ground、红色
cube、灯光和 `320x240` Camera，取得非空 RGBA 帧（shape `240x320x4`、std `18.106`），
并正常关闭。该 smoke 验证 Kit、RTX 实时渲染和相机链路，不冒充 VLN episode 验证。

环境中间件使用 InternUtopia 2.2.1 源码 overlay，固定 commit
`b0a9520c586317c2743023c153cbf7c4f04f4732`。没有直接安装其发行包，因为其
`transformers==4.26.1` 固定依赖与三个 VLN 已验证的 `4.51.0` 冲突；其余直接运行依赖均
锁定在 `config/conda/harnessvln.yaml`，当前 `pip check` 通过。InternNav 与 VLNVerse 分别
固定 commit `7a5c62400ac45b313d9b709c740b64191556a242` 和
`d444c0412ac50583b0162f690edc0ce1b2aa8639`。

`envs.internutopia` 从一个 `BenchmarkCase` 构造一个官方原生 episode：VLN-PE 使用 MP3D
坐标/四元数转换与 H1 offset，VLNVerse 保留 Kujiale 坐标；之后分别调用上游
`get_config` 和 `generate_episode`，再交给各自 extension 与 InternUtopia `Env`。两条路径
已用临时 USD/H1 资源真实生成 list-of-one native config，path key 分别为 `99999_1` 和
`route_2`。当前专用场景、H1 USD/locomotion policy 和 episode 尚未落地，所以只声明
`data_contract`，不声明完成真实 VLN-PE/VLNVerse reset。配置见
`config/benches/vln_pe.yaml`、`config/benches/vlnverse.yaml` 和对应 `config/envs/` 文件。

Kit 的 SimulationApp 是进程级资源，两套环境组件都声明 `serial: true`；Runner 仍可并行
完整任务，但 Isaac 的扩展方式是启动多个隔离环境服务进程，而不是在一个 Kit 进程中并发
创建多个生命周期。
