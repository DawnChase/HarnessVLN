# 统一运行环境

项目只使用一个 Conda 环境 `harnessvln`，Python 固定为 3.10。Agent、VLN worker、
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

StreamVLN 已用同一环境完成本地离线加载与单帧 RGB-D 推理：主 checkpoint revision 为
`f1f76c66083c362ddfcd2610167f9c4e4a46c027`，四个 safetensors 分片覆盖索引中的
764 个 tensor；SigLIP revision 为 `9fdffc58afc957d1a03a25b10dba0329ab15c2a3`。
实际构造得到 8,030,345,248 个参数、残留 meta 参数为 0，测试输入解析出动作
`(turn_right, turn_right, turn_right, turn_right)`。运行参数见
`config/vln/streamvln.yaml`。

JanusVLN Base 也已在该环境离线加载并完成单帧 RGB 推理。ModelScope checkpoint revision
固定为 `33f932a4ea6bdc34afca9f5b79a8b4537cd02509`；四个分片的 SHA-256 与官方清单一致，
2068 个 tensor 完整覆盖索引。实际模型包含 9,314,832,390 个参数和 24 层 VGGT，残留
meta 参数为 0，测试输入生成 `TURN_RIGHT`。为保持官方 Transformers 4.50 的图像处理
行为，统一环境中的 adapter 显式设置 `use_fast=False`；运行参数见
`config/vln/janusvln.yaml`。

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

已在 MP3D R2R `val_unseen` episode 1 验证真实 reset、RGB-D render、navmesh、前进和转向：
RGB 为 `480x640x3`，Depth 为 `480x640x1`，前进一步位移约 0.25 m。该结果只确认
Habitat/R2R 环境链路；完整 split 评分、GOAT 与三个模型的 episode 对照仍按 release gate
分别验收。

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
