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

Habitat、AI2-THOR 和 Isaac Sim SDK 后续也只加入 `harnessvln`；加入一种 SDK 后必须重新
执行 `pip check`、全量单测及对应模拟器 smoke test，配置文件同步追加精确版本。
