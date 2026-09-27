# Stardew Valley 视觉识别 + PPO 自动钓鱼助手

Windows 上的实验性钓鱼助手：从游戏画面定位小游戏，用视觉 CNN 测量鱼、绿条与捕获进度，再由冻结的 PPO 策略决定鼠标左键的按住或松开。F1 模式还会循环抛竿、识别咬钩、提竿与收获；不读取或修改游戏内存，也不直接设置钓鱼条坐标。

本仓库是 [lovida8254/stardew_autofish](https://github.com/lovida8254/stardew_autofish) 的衍生项目，保留原有辅助程序与 Git 历史。旧版说明见 [README.legacy.ko.md](README.legacy.ko.md)；以下是当前视觉 + PPO 版本。

## 一键启动

1. 在 Windows 10/11 上安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)（推荐，负责下载 Python 3.12 与加速安装依赖），或自行安装 Python 3.12。准备支持 CUDA 的 NVIDIA 显卡与驱动。
2. 下载仓库完整目录后双击 **`启动视觉PPO助手.bat`**。首次启动会创建 `training/.venv-vision`，按 [requirements-vision-gui-lock.txt](requirements-vision-gui-lock.txt) 安装依赖，然后检查 CUDA、模型与素材并打开 GUI。首次下载 PyTorch 可能较慢，之后无需重复安装。
3. 运行星露谷物语，切到游戏窗口，在水边站好、选中鱼竿并把鼠标指向投点。按 **F1** 开始全自动；按钮启动会给 3 秒切回游戏。再次按 F1 或按 **F8** 停止。

若已有其他钓鱼助手，先停止它，避免两个程序同时控制鼠标。也可只点 GUI 的“开始”让用户手动抛竿、上钩，程序仅接管小游戏。GUI 可在停止状态下选择“只观察”、手动框选完整钓鱼面板，或调整等级、渔具和鱼中心偏移。

| 操作 | 用途 |
|---|---|
| F1 | 在游戏前台启动或停止全自动循环；默认蓄力按住 1.05 秒 |
| F8 | 停止并释放程序按住的鼠标 |
| F6 | 框选当前游戏画面中的完整钓鱼面板 |
| F9 | 保存最近的画面、时间戳与动作诊断片段到 `logs/` |

默认上钩方式为局部画面确认。音频辅助和鱼竿自动上钩附魔可在 GUI 中切换。GUI 找不到小游戏或追踪异常时可先按 F9，再查看预览和日志。切出游戏、移动或打开菜单会停止全自动循环。程序不会自行走到水边或管理背包。

命令行诊断（只加载模型并用样本画面做一次推理，不向游戏发送输入）：

```powershell
.\start_vision_ppo.ps1 -CheckOnly
```

需要仅安装依赖时用 `-InstallOnly`，需要重新安装时加 `-ForceInstall -InstallOnly`。如果 CUDA 检查失败，先检查 NVIDIA 驱动与所装 PyTorch 是否为 `+cu128` 构建。当前视觉运行器要求 CUDA；没有 NVIDIA GPU 时可以单独运行 CPU 钓鱼模拟器。

## 模型和训练过程

| 组件 | 当前实现 |
|---|---|
| 数值模拟 | 参照 [Pufferdle 固定版本](training/SOURCES.md)实现鱼与绿条运动；60 Hz 物理，30 Hz 策略决策；55 种鱼、五类运动。模拟器支持等级 0–20 与五种渔具。 |
| PPO 控制器 | Stable-Baselines3 PPO；16 帧 × 每帧 18 个观测值 + 6 个配置值，共 294 维；Actor/Critic 各 `[128, 128]` 的 MLP。输出 0=松开、1=按住。无注意力层，也没有单独的鱼种分类器。 |
| PPO 训练 | 先用课程训练约 200 万决策，再在完整鱼池续训到 **10,001,216** 决策；训练采样覆盖有效钓鱼等级 0–10、无渔具。权重为 `training/runs/20260927_scaling_10m/best.zip`。 |
| 视觉 CNN | 输入 RGB `192×640`；四尺度 U-Net 风格 CNN，通道 `16/32/64/96`，约 38.7 万参数、**0 层注意力**。预测六类像素分割、四个纵向坐标和四个存在分数。训练 6,000 次优化更新，batch 12，即 72,000 次合成样本呈现；权重为 `training/vision_runs/20260927_v1_synthetic/best.pt`。 |
| 实时连接 | OpenCV 粗定位 + CNN 几何测量 + 16 帧观测编码 + PPO；只通过鼠标输入控制。根据绿条长度估计有效等级，可在 GUI 覆盖。 |

观测包括鱼中心、绿条中心和高度、基于历史的速度估计、捕获进度及变化、上一动作、动作切换间隔、帧间隔、可见性/有效性、画面年龄和剩余时间。模型没有获得鱼种 ID、鱼的隐藏目标位置或未来随机数。渔具已进入模拟器与观测接口，但**此版权重并未在非“无渔具”配置上训练**。真实游戏中的等级估计、图像误差与输入延迟也会影响效果。

### 已有评估

| 指标 | 结果与适用范围 |
|---|---|
| PPO 总捕获率 | **88.94%**；独立预留的 5,000 个**模拟**种子，等级 0–10、无渔具、理想几何输入。起点约 200 万步为 86.42%。 |
| 难度 ≥90 | **32.37%**，同一模拟预留集中的 519 次。 |
| 等级 0 | **81.88%**，同一模拟预留集中的 469 次。 |
| 视觉鱼中心误差 | P95 **0.67 px**；256 张合成验证图上的模型输入画布像素，不是真实游戏画面精度。 |
| 视觉推理耗时 | 在 RTX 4070 SUPER 上，单张已裁剪画面 P95 **4.73 ms**；不含截图、面板搜索、PPO 或鼠标输入。 |

以上成功率不是端到端游戏捕获率。真实游戏已有人工场景试用，但还没有足够的系统化标注与独立试验来报告可信的实机捕获率。视觉验证与 PPO 评估也来自不同实验，不能把两项指标相乘得到实机准确率。训练仅使用一个 PPO 随机种子；下一步值得补充更多真实画面标注、难鱼场景与渔具训练。详细数值见 [1000 万步报告](training/runs/20260927_scaling_10m/SCALING_REPORT.md)、[视觉验证数据](training/vision_runs/20260927_v1_synthetic/validation_006000.json) 和 [训练说明](training/README.md)。

## 复现与目录

- `vision_ppo_gui.py` / `vision_live.py` / `auto_fishing.py`：GUI、画面适配器和 F1 自动循环。
- `training/fishing_sim/`：数值物理、Gymnasium 环境、294 维观测、训练与评估。
- `training/fishing_vision/`：合成渲染、CNN 训练和推理。
- `training/runs/20260927_scaling_10m/`、`training/vision_runs/20260927_v1_synthetic/`：发布的最佳权重及最少量的指标/清单；完整中间检查点与用户录屏未上传。
- `training/vision_data/assets/`、`auto_assets/`：运行需要的小型图像模板；来源见 [training/SOURCES.md](training/SOURCES.md) 与各清单。

运行 CPU 模拟器：双击 **`启动训练模拟器.bat`**，或在 `training/` 执行 `./setup.ps1` 后运行 `./.venv/Scripts/python.exe -m fishing_sim.serve --port 8767`，访问 `http://127.0.0.1:8767/`。新训练请指定**新的** `--run-dir`，例如：

```powershell
cd training
.\.venv\Scripts\python.exe -m fishing_sim.train --steps 2000000 --seed 42 --run-dir runs/my_2m
.\.venv\Scripts\python.exe -m fishing_sim.train --steps 7997056 --resume runs/my_2m/final.zip --no-curriculum --seed 42 --run-dir runs/my_10m
.\.venv-vision\Scripts\python.exe -m fishing_vision.train --assets vision_data/assets --run vision_runs/my_vision --steps 6000 --batch-size 12 --seed 92701
```

这些命令说明训练阶段；精确复刻已发布指标还需要原始负样本集合、完整中间检查点、相同软件/随机环境及预设评估种子。视觉训练使用 CUDA 环境，数值 PPO 训练使用 CPU 环境。固定来源、数值模型与有意差异记在 [training/SOURCES.md](training/SOURCES.md)。

## 来源与发布说明

原始助手代码来自 [lovida8254/stardew_autofish](https://github.com/lovida8254/stardew_autofish)。钓鱼动力学和若干训练素材参照 [abmasud1214/pufferdle](https://github.com/abmasud1214/pufferdle) 的固定提交；游戏图像权利属于各自权利人。本仓库没有添加覆盖这些上游代码或素材的统一许可证，也不声称与所有星露谷物语版本完全一致。公开仓库不含个人录像、诊断日志、虚拟环境或本机配置。
