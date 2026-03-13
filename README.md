# LeRobotPro Songling ALOHA 完整实战文档

本仓库主 README 已切换为 Songling ALOHA（松灵 ALOHA）专项文档，覆盖从仓库下载到环境配置、端口配置、示教验证、数据采集、训练的完整流程。

适用硬件模型（当前文档假设）：
- 左右各一套 Songling 集成主从链路（CAN）
- 三路 UVC 相机：`left_high`、`left_elbow`、`right_elbow`
- 无底盘（v1）

核心事实（非常重要）：
- 该硬件是“集成主从链路”，上电后硬件已完成跟随关系。
- 当前推荐路径是“原始 CAN + 三相机采集/可视化”，不是通用软件 leader/follower 控制路径。
- 统一配置文件是 `examples/songling_aloha/teleop.yaml`。

---

## 1. 仓库下载与目录

### 1.1 克隆仓库

```bash
git clone https://github.com/yingsanyi/lerobotPro.git
cd lerobotPro
```

如果你已在本地仓库中，可直接跳过。

### 1.2 关键目录

```text
examples/songling_aloha/
  teleop.yaml                  # 单一配置源（CAN/相机/数据集/语音）
  record_compat.py             # 推荐采集入口（兼容 lerobot-record 参数）
  record_raw_can_dataset.py    # 原始 CAN 采集实现
  visualize_teleop_live.py     # 实时可视化（Rerun）
  capture_three_cameras.py     # 三相机拍照验证
  sniff_can_frames.py          # CAN 抓包
```

---

## 2. 环境配置

### 2.1 Conda 环境

你可以使用已有环境，也可以新建一个自定义环境名。

方案 A：使用你已有的环境（示例名 `lerobot_v050`）：

```bash
conda activate lerobot_v050
cd /home/aiteam/wyl/lerobot0.5.0/lerobotPro
pip install -e .
# 如需 OpenArm/Songling 相关依赖，再执行下面这行
pip install -e ".[openarms]"
pip check
python -c "import lerobot; print(lerobot.__version__)"
```

方案 B：新建环境（环境名可自定义）：

```bash
ENV_NAME=lerobot_songling
conda create -n ${ENV_NAME} python=3.12 -y
conda activate ${ENV_NAME}
cd /path/to/lerobotPro
pip install -e .
# 如需 OpenArm/Songling 相关依赖，再执行下面这行
pip install -e ".[openarms]"
pip check
python -c "import lerobot; print(lerobot.__version__)"
```

说明：
- `pip install -e .` 是源码可编辑安装（开发模式）。
- `pip install -e ".[openarms]"` 会额外安装 OpenArm/Songling 常用依赖。

### 2.2 可选依赖

### A. 语音（Piper）

本仓库默认已支持从本地路径发现 Piper：
- `third_party/piper/piper/piper`
- `third_party/piper/models/*.onnx`

`teleop.yaml` 当前默认：
- `voice_engine: piper`
- `voice_lang: zh`

### B. Hugging Face（仅当你要上传数据/模型）

```bash
huggingface-cli login
```

### C. Weights & Biases（仅当你要在线训练日志）

```bash
wandb login
```

---

## 3. 端口配置（CAN + 相机）

### 3.1 CAN 口拉起与测试

```bash
lerobot-setup-can --mode=setup --interfaces=can0,can1 --use_fd=false --bitrate=1000000
lerobot-setup-can --mode=test --interfaces=can0,can1 --use_fd=false --bitrate=1000000
```

如果网卡名不是 `can0/can1`，请替换为实际名称。

### 3.2 自动识别左右链路并写回配置

先识别左侧：

```bash
lerobot-find-port \
  --kind can \
  --can-mode traffic \
  --interfaces can0,can1 \
  --traffic-direction rx \
  --traffic-window-s 3 \
  --min-msgs 5 \
  --settle-time-s 1.0 \
  --write-songling-config examples/songling_aloha/teleop.yaml \
  --songling-side left
```

再识别右侧：

```bash
lerobot-find-port \
  --kind can \
  --can-mode traffic \
  --interfaces can0,can1 \
  --traffic-direction rx \
  --traffic-window-s 3 \
  --min-msgs 5 \
  --settle-time-s 1.0 \
  --write-songling-config examples/songling_aloha/teleop.yaml \
  --songling-side right
```

### 3.3 相机探测与写回配置

先列出设备：

```bash
lerobot-find-cameras opencv
```

保存快照辅助映射：

```bash
lerobot-find-cameras opencv \
  --mode snapshot \
  --output-dir /tmp/lerobot_camera_probe \
  --opencv-width 640 \
  --opencv-height 480 \
  --opencv-fourcc MJPG \
  --opencv-backend 200 \
  --snapshot-name-style short
```

将三路相机写回 `teleop.yaml`（按路径）：

```bash
lerobot-find-cameras opencv \
  --mode configure-songling \
  --write-songling-config examples/songling_aloha/teleop.yaml \
  --songling-left-high /dev/v4l/by-id/<TOP_CAMERA>-video-index0 \
  --songling-left-elbow /dev/v4l/by-id/<LEFT_ELBOW_CAMERA>-video-index0 \
  --songling-right-elbow /dev/v4l/by-id/<RIGHT_ELBOW_CAMERA>-video-index0 \
  --opencv-width 640 \
  --opencv-height 480 \
  --opencv-fps 30 \
  --opencv-fourcc MJPG \
  --opencv-backend 200
```

或按 manifest 索引写回：

```bash
lerobot-find-cameras opencv \
  --mode configure-songling \
  --write-songling-config examples/songling_aloha/teleop.yaml \
  --songling-manifest /tmp/lerobot_camera_probe/manifest.json \
  --songling-left-high-index 2 \
  --songling-left-elbow-index 0 \
  --songling-right-elbow-index 1
```

### 3.4 可选：原始 CAN 抓包

当你要做协议分析或二次适配时，先抓包：

```bash
python examples/songling_aloha/sniff_can_frames.py \
  --interfaces can0 can1 \
  --duration-s 20 \
  --bitrate 1000000 \
  --output-dir /tmp/songling_can_sniff
```

---

## 4. 示教前验证（强烈推荐）

### 4.1 三相机拍照验证

```bash
python examples/songling_aloha/capture_three_cameras.py \
  --config-path examples/songling_aloha/teleop.yaml \
  --output-dir /tmp/songling_snapshots
```

### 4.2 实时可视化（原始 CAN + 3 路相机）

```bash
python examples/songling_aloha/visualize_teleop_live.py \
  --config-path examples/songling_aloha/teleop.yaml \
  --raw-can-mode
```

低负载模式：

```bash
python examples/songling_aloha/visualize_teleop_live.py \
  --config-path examples/songling_aloha/teleop.yaml \
  --raw-can-mode \
  --can-poll-max-msgs 64 \
  --rerun-log-fps 5
```

远程 Rerun 服务：

```bash
# 终端 A
rerun --serve-web --port 9876 --web-viewer-port 9090

# 终端 B
python examples/songling_aloha/visualize_teleop_live.py \
  --config-path examples/songling_aloha/teleop.yaml \
  --raw-can-mode \
  --display_ip 127.0.0.1 \
  --display_port 9876
```

---

## 5. 数据采集（示教）

推荐入口：`examples/songling_aloha/record_compat.py`。

它会：
- 自动读取 `teleop.yaml`
- 统一 `lerobot-record` 参数风格
- 在检测到集成链路配置时自动转发到 `record_raw_can_dataset.py`
- 继承语音配置（`play_sounds`、`voice_*`）

### 5.1 本地默认采集（不需要 HF）

```bash
python examples/songling_aloha/record_compat.py \
  --config_path=examples/songling_aloha/teleop.yaml \
  --dataset.repo_id=local/songling_aloha_demo \
  --dataset.single_task="Bimanual teleoperation with Songling ALOHA profile" \
  --dataset.push_to_hub=false
```

如果不显式传 `--dataset.root`：
- 优先使用 `teleop.yaml` 中的 `dataset.root`
- 若未配置则默认到 `<repo>/outputs/songling_aloha`

### 5.2 指定固定目录采集（你当前常用方式）

```bash
python examples/songling_aloha/record_compat.py \
  --config_path=examples/songling_aloha/teleop.yaml \
  --dataset.repo_id=local/songling_aloha_run_001 \
  --dataset.root=/home/aiteam/wyl/lerobot0.5.0/lerobotPro/outputs/songling_aloha_run_001 \
  --resume=false \
  --dataset.auto_increment_root=false \
  --dataset.push_to_hub=false
```

### 5.3 续录同一目录

```bash
python examples/songling_aloha/record_compat.py \
  --config_path=examples/songling_aloha/teleop.yaml \
  --dataset.repo_id=local/songling_aloha_run_001 \
  --dataset.root=/home/aiteam/wyl/lerobot0.5.0/lerobotPro/outputs/songling_aloha_run_001 \
  --resume=true \
  --dataset.auto_increment_root=false
```

### 5.4 本地采集 + 同步上传到 HF

先登录 HF：

```bash
huggingface-cli login
```

再采集：

```bash
python examples/songling_aloha/record_compat.py \
  --config_path=examples/songling_aloha/teleop.yaml \
  --dataset.repo_id=your_hf_username/songling_aloha_demo \
  --dataset.root=outputs/songling_aloha \
  --dataset.push_to_hub=true
```

### 5.5 语音参数覆盖（可选）

```bash
python examples/songling_aloha/record_compat.py \
  --config_path=examples/songling_aloha/teleop.yaml \
  --dataset.repo_id=local/songling_aloha_demo \
  --dataset.push_to_hub=false \
  --play_sounds=true \
  --voice_lang=zh \
  --voice_engine=piper \
  --voice_piper_model=third_party/piper/models/zh_CN-huayan-medium.onnx \
  --voice_piper_binary=third_party/piper/piper/piper
```

### 5.6 直接调用原始采集脚本（高级）

```bash
python examples/songling_aloha/record_raw_can_dataset.py \
  --config_path=examples/songling_aloha/teleop.yaml \
  --dataset.repo_id=local/songling_aloha_demo \
  --dataset.root=outputs/songling_aloha \
  --dataset.push_to_hub=false
```

---

## 6. 采集后检查与可视化

### 6.1 检查数据目录

```bash
ls -la outputs/songling_aloha
ls -la outputs/songling_aloha/meta
```

### 6.2 数据集可视化

```bash
lerobot-dataset-viz \
  --repo-id local/songling_aloha_demo \
  --root outputs/songling_aloha \
  --episode-index 0
```

如果你采集在固定目录：

```bash
lerobot-dataset-viz \
  --repo-id local/songling_aloha_run_001 \
  --root /home/aiteam/wyl/lerobot0.5.0/lerobotPro/outputs/songling_aloha_run_001 \
  --episode-index 0
```

---

## 7. 训练（ACT）

下面给出三套常用版本。
建议把 `--dataset.repo_id` 写成你采集时使用的同一个 ID，便于日志与产物管理。

### 7.1 纯本地测试版（不需要 HF，不需要 wandb）

```bash
HF_HUB_OFFLINE=1 WANDB_DISABLED=true lerobot-train \
--policy.type=act \
--dataset.repo_id=local/songling_aloha_run_001 \
--dataset.root=/home/aiteam/wyl/lerobot0.5.0/lerobotPro/outputs/songling_aloha_run_001 \
--policy.push_to_hub=false \
--policy.device=cuda \
--output_dir=outputs/train/local_act_songling_aloha_test \
--job_name=local_act_songling_aloha_test \
--batch_size=32 \
--steps=2000 \
--save_freq=500 \
--num_workers=0 \
--wandb.enable=false
```

### 7.2 本地正式训练（不上传 HF，不用 wandb）

```bash
HF_HUB_OFFLINE=1 WANDB_DISABLED=true lerobot-train \
--policy.type=act \
--dataset.repo_id=local/songling_aloha_run_001 \
--dataset.root=/home/aiteam/wyl/lerobot0.5.0/lerobotPro/outputs/songling_aloha_run_001 \
--policy.push_to_hub=false \
--policy.device=cuda \
--output_dir=outputs/train/act_songling_aloha_run_001 \
--job_name=act_songling_aloha_run_001_training \
--batch_size=32 \
--steps=300000 \
--save_freq=10000 \
--num_workers=0 \
--wandb.enable=false
```

### 7.3 在线训练日志 + 推送模型（HF + wandb）

先登录：

```bash
huggingface-cli login
wandb login
```

再启动训练：

```bash
lerobot-train \
--policy.type=act \
--dataset.repo_id=YSanYi/songling_aloha_run_001 \
--dataset.root=/home/aiteam/wyl/lerobot0.5.0/lerobotPro/outputs/songling_aloha_run_001 \
--policy.push_to_hub=true \
--policy.repo_id=YSanYi/act_songling_aloha_run_001_policy \
--policy.device=cuda \
--output_dir=outputs/train/act_songling_aloha_run_001 \
--job_name=act_songling_aloha_run_001_training \
--batch_size=32 \
--steps=300000 \
--save_freq=10000 \
--num_workers=0 \
--wandb.enable=true \
--wandb.project=act_training_songling_aloha_run_001 \
--wandb.notes="act training on local Songling ALOHA dataset."
```

---

## 8. 常见问题（FAQ）

### 8.1 `wandb.errors.UsageError: No API key configured`

原因：启用了 `--wandb.enable=true` 但未登录。

解决方案二选一：

```bash
# 方案 A：关闭 wandb
--wandb.enable=false
```

```bash
# 方案 B：登录后继续
wandb login
```

### 8.2 不想依赖 HF，但训练/加载仍访问网络

使用：

```bash
HF_HUB_OFFLINE=1
```

并配合：

```bash
--policy.push_to_hub=false
```

### 8.3 `draccus is required ...`

说明当前环境依赖不完整，请确认在正确 conda 环境，并重新安装：

```bash
conda activate lerobot_v050
pip install -e ".[openarms]"
```

### 8.4 `Dataset root already exists` 相关错误

如果你想在原目录续录：

```bash
--resume=true --dataset.auto_increment_root=false
```

如果你想新开目录：

```bash
--resume=false --dataset.auto_increment_root=true
```

或直接改 `--dataset.root`。

### 8.5 `ResourceTracker ... _recursion_count` 异常

建议先用最稳参数验证：

```bash
--num_workers=0
```

对于采集脚本，本仓库已包含对该问题的兼容处理。

---

## 9. 推荐执行顺序（最短路径）

1. 激活环境并安装依赖。  
2. `lerobot-setup-can` 配置 CAN。  
3. `lerobot-find-port` + `lerobot-find-cameras` 写回 `teleop.yaml`。  
4. `capture_three_cameras.py` 验证相机画面。  
5. `visualize_teleop_live.py --raw-can-mode` 验证链路实时状态。  
6. `record_compat.py` 录制数据（先本地版）。  
7. `lerobot-dataset-viz` 抽查样本。  
8. 跑“纯本地测试版训练”确认流程通，再上正式训练参数。  

---

## 10. 安全提示

- 首次联调请从小幅度、单关节动作开始。
- 始终确保急停可达。
- 在未充分验证前，不要放大速度和目标位移。
