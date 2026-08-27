# 数据、模型与环境

Harness 代码保持轻量，大型数据、场景、checkpoint 与上游源码按职责放在独立目录，并由 YAML
记录固定版本。

## 统一 Conda 环境

```bash
conda env create -f config/conda/harnessvln.yaml
conda activate harnessvln
```

环境名固定 `harnessvln`，Python 固定 3.10，满足“3.10 以上”的代码基线，同时适配当前模拟器
版本。Agent、worker、Bench 和中间件即便跨进程，也从同一环境启动。

当前模型验证组合使用 Torch 2.8.0 + CUDA 12.8 和 FlashAttention 2.8.3；具体安装顺序与
Habitat/Isaac 的详细安装记录保存在本地项目资料中。版本调整应先跑真实模型加载与固定 trace，
不能只以 `pip check` 为准。

## 目录约定

```text
data/
  datasets/          # episode 与标注
  scene_datasets/    # MP3D、HM3D、Gibson 等共享场景
  assets/            # GOAT cache、H1 USD/policy 等配套资源
model/               # StreamVLN / JanusVLN / DualVLN checkpoint
cache/upstream/      # Habitat、模型、InternUtopia 等固定源码 checkout
runs/                # 本地结果和 Memory 文件
```

已有大型资源可通过软链接接入；Bench 和 Environment 使用稳定的项目内逻辑路径，不在代码中硬编码
外部磁盘挂载点。

## 当前已有数据

| 数据 | 状态 | 主要规模/用途 |
|---|---|---|
| R2R-CE | 已落地 | val_unseen 1839；三个 VLN 的当前真实验证集 |
| RxR / EnvDrop | 已落地 | 后续 VLN 扩展数据，当前未有 Bench adapter |
| GOAT-Bench HM3D v1 | 已落地 | 复合 episode，validation split 与目标 cache |
| ObjectNav MP3D v1 | 已落地 | val 2195 |
| ObjectNav HM3D v2 | 已落地 | val 1000 |
| RoboTHOR ObjectNav 2021 | 已落地 | val 1800，场景由 Unity build 提供 |
| MP3D / HM3D / Gibson scenes | 已落地 | 多 Bench 共享，不复制 |
| VLN-PE 专用资源 | 缺失 | MP3D-PE scene、episode、H1 资产/policy |
| VLNVerse 专用资源 | 缺失 | Kujiale scene、formal split、H1 资产 |

详细本地路径、hash 与文件数保存在本地项目资料中。资源存在与否应以配置所需路径和 loader
contract 检查为准，空目录不算已落地。

## 三个模型资源

模型 YAML 分别固定：

- upstream source root 与 commit；
- 主 checkpoint 路径和 revision；
- tokenizer / vision tower / depth checkpoint 等附属 revision；
- worker command、cwd、环境变量与离线标志；
- device、history/frame 数、最大步数和超时。

权重放在 `model/`，不混入 `data/`。上游源码放 `cache/upstream/`，不复制进 `src/vln/`；项目子目录
只保存适配代码和必要的模型胶水。

## 模拟器固定版本

| 系统 | 当前固定 | 注意事项 |
|---|---|---|
| Habitat-Sim/Lab | 0.3.3 + 固定 commit | headless 源码构建；R2R/ObjectNav/GOAT |
| AI2-THOR | 2.7.2 + RoboTHOR build commit | 当前硬件使用 Xvfb + Mesa GLX |
| Isaac Sim | 4.5.0 | pip index、Kit cache、进程级串行资源 |
| InternUtopia | 2.2.1 source overlay | 避免其旧 transformers pin 覆盖统一环境 |

## 新机器验收顺序

1. `pytest -q` 跑无大型资源测试；
2. dummy CLI 生成 Manifest；
3. 检查 dataset loader contract 和配置路径；
4. 分别做模拟器 reset/render/action/stop smoke；
5. 单模型离线加载与单帧推理；
6. `max_cases: 1` 完整 Harness 链路；
7. 固定三例 run-scope trace；
8. 最后跑完整 split 与官方 evaluator parity。

按层验收能把依赖故障、环境故障、模型故障和 Harness 故障分开。
