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
分别验收。AI2-THOR 和 Isaac Sim SDK 也只加入 `harnessvln`；加入后重新执行 `pip check`、
全量单测及对应模拟器 smoke test，并同步固定版本。
