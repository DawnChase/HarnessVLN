# HarnessVLN 数据布局

项目中的 `data` 最终解析到 `/home2/csl2/Dataset/HarnessVLN`。已有数据只建立软链接；
本项目新下载的数据直接保存在该目录中。Episode 数据、公共场景和模型配套资源分别放在
`datasets/`、`scene_datasets/` 和 `assets/`，Bench 不复制公共场景。

```text
data/
  datasets/
    r2r -> PointVLN/datasets/r2r
    rxr -> PointVLN/datasets/rxr
    envdrop -> PointVLN/datasets/envdrop
    goat_bench -> RouterNav/datasets/goat_bench
    objectnav/                       # 本地实体
      mp3d/v1/
      hm3d/v2/
    robothor-objectnav-2021/         # 本地实体
  scene_datasets/
    mp3d -> PointVLN/scene_datasets/mp3d
    hm3d -> PointVLN/scene_datasets/hm3d/hm3d
    gibson -> PointVLN/scene_datasets/gibson
    habitat-test-scenes -> PointVLN/scene_datasets/habitat-test-scenes
  assets/
    goat_bench/                      # 本地实体；goal cache 与 checkpoint
```

## 当前可用数据

| 路径 | 形态 | 内容与校验 |
|---|---|---|
| `datasets/r2r` | 软链接 | R2R-CE：train 10819、val_seen 778、val_unseen 1839、test 3408；引用的 90 个 MP3D 场景齐全 |
| `datasets/rxr` | 软链接 | RxR follower/guide 的 train、val_seen、val_unseen、test_challenge；规范 gzip 文件通过校验 |
| `datasets/envdrop` | 软链接 | EnvDrop train：146304 个 episode，GT 数量一致 |
| `datasets/goat_bench` | 软链接 | GOAT-Bench HM3D v1：train 680000 个复合 episode；三个 validation split 各 360 个 episode |
| `datasets/objectnav/mp3d/v1` | 实体 | Habitat ObjectNav：train 2632422、val 2195、val_mini 30 |
| `datasets/objectnav/hm3d/v2` | 实体 | Habitat ObjectNav：train 7196434、val 1000、val_mini 30 |
| `datasets/robothor-objectnav-2021` | 实体 | debug/train/val/test：4/108000/1800/2040 个 episode |
| `scene_datasets/*` | 软链接 | MP3D 90、HM3D 900、Gibson 492、Habitat test scenes 3；ObjectNav 引用覆盖为 MP3D 67/67、HM3D 181/181 |
| `assets/goat_bench` | 实体 | GOAT goal cache 与两个官方 policy checkpoint；4.5 GB、1311 个文件，固定 revision dry-run 待下载数为 0 |

GOAT 的一个复合 episode 含 5--10 个 goal。现有四个 split 的 gzip 均通过校验，
train/val_seen/val_seen_synonyms/val_unseen 分别含 680000/360/340/360 个 episode，
对应 5098248/2757/2525/2669 个 goal。
GOAT cache 四个 split 分别含 435/216/144/504 个文件；两个 checkpoint 大小为
77777474 和 75672275 bytes，并通过 PyTorch zip 容器检查。

## 固定来源

- GOAT 配套资源：[`axel81/goat-bench`](https://huggingface.co/datasets/axel81/goat-bench/tree/6df4daf962da14d5a57315e9c80f2969f814e6d9)，revision `6df4daf962da14d5a57315e9c80f2969f814e6d9`。
- Habitat ObjectNav MP3D v1：[`objectnav_mp3d_v1.zip`](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip)，SHA-256 `b1c9cd40bd94043705e20539034254a0bf8c49dd405d93dab0cef75ae2fd5bdf`。
- Habitat ObjectNav HM3D v2：[`objectnav_hm3d_v2.zip`](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip)，SHA-256 `f551a0d8560804dc385c52f9aedb646c70917da36c1902d848b4fbc1615515d0`。
- RoboTHOR ObjectNav 2021：[`robothor-challenge`](https://github.com/allenai/robothor-challenge/tree/f3c4f35bc397aff4d5236e269efe3ff41f6d218e)，commit `f3c4f35bc397aff4d5236e269efe3ff41f6d218e`；归档 SHA-256 `80b552dd77f5abc5196909cd238ab0aac892c175c17e50104008a97c57d3db5d`。

## 当前未落地

VLN-PE、VLNVerse 的专用场景、episode 和 checkpoint 尚未放入 `data`。StreamVLN、
JanusVLN、DualVLN 的模型权重已按职责存放在 `model`，不复制到数据目录。
AI2-THOR/RoboTHOR 场景由对应 simulator 包管理，也不在这里复制。
