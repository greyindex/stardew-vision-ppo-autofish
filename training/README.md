# 离线钓鱼模拟与 PPO 训练

双击项目目录的 **`启动训练模拟器.bat`**，或访问已经启动的 [训练页面](http://127.0.0.1:8767/)。无需打开游戏。

公开仓库附带首轮 `best.zip` / `final.zip`、1000 万步的 `best.zip`、实验清单及汇总报告；其余中间检查点、逐回合原始评估和本机日志未上传。下文提到的 `latest.zip`、`initial.zip`、`checkpoint_*.zip` 是本地训练时产生的文件。

## 现在能做什么

- 选择 55 种鱼、钓鱼等级（0–20）、五种渔具配置，观看捕获过程。
- 切换 MPC、简单预测规则、PPO 最佳模型或手动控制；手动模式在模拟画布内按住鼠标，或按住空格。
- 新建训练、从最近检查点继续训练、保存并停止；页面展示进度与固定开发集的成功率曲线。
- 固定随机种子重放轨迹；命令行批量训练和评估；保存可重载的模型、日志、逐次评估及依赖版本。

当前模型训练范围是**无渔具、等级 0–10、目录中的全部鱼**。界面也允许试验其他渔具及更高等级，但首轮 PPO 没有接受这些配置的训练。MPC 可以直接使用这些渔具的已实现物理模型。

## 首轮模型

目录：`runs/20260927_ppo_v1/`。`best.zip` 是按固定开发集的捕获率选择的模型；`latest.zip` / `final.zip` 是末尾权重；`initial.zip` 是初始权重。第一次训练约 200 万个决策，分为基础追踪、增加难度、完整鱼池三阶段。每个决策执行两个物理帧。

首轮实测与未参与选模的保留测试结果见 `FIRST_RUN.md`。页面上的学习曲线使用固定的开发集，不能把它当作每次全新的最终测试。

## 1000 万步续训对照

`runs/20260927_scaling_10m/` 从首轮末尾的 2,002,944 步权重与优化器继续，累计目标 10,000,000 步。完整鱼池、等级 0–10、无渔具、网络、奖励、学习率及 30 Hz 决策均保持一致；完成过的简单课程不再重复。

- 预先指定的条件：`experiment_plan.json`。
- 固定 2,000 次开发评估：从起点开始，每增加约 100 万步记录一次；终点在最后一次优化器更新后另做评估。
- 独立 5,000 次保留测试：比较起点、终点及按开发集选择的模型。
- 训练结束并完成分析后：`SCALING_REPORT.md`、`scaling_curve.png`、`scaling_curve.csv`、`scaling_summary.json`。
- 每个评估节点保留 `checkpoint_*.zip`；`best.zip` 按开发集选择，`final.zip` 为训练终点。

这是一条训练种子的续训轨迹；评估的置信区间不涵盖不同训练种子之间的差异。

复现命令（指定尚不存在的实验目录）：

```powershell
.\.venv\Scripts\python.exe -m fishing_sim.train --steps 7997056 --resume runs/20260927_ppo_v1/final.zip --no-curriculum --seed 42 --eval-every 1000000 --eval-episodes 2000 --eval-seed 3000000 --eval-at-start --eval-at-end --save-checkpoints --run-dir runs/NEW_SCALING_RUN

# 本轮训练完成后执行保留测试及生成报告；条件来自实验目录内的 experiment_plan.json
uv pip install --python .venv/Scripts/python.exe -r requirements-analysis.txt
.\.venv\Scripts\python.exe -m fishing_sim.scaling_report --run-dir runs/20260927_scaling_10m --evaluate-holdout
```

注意 CLI 的 `--steps` 表示**新增步数**。本轮新增请求 7,997,056 步；训练以完整 rollout 为单位，因此累计实际步数会略超 1000 万。

## 环境和观测

`FishingEnv.reset(seed=..., options=...)` 返回 `(obs, info)`；`step(action)` 返回 `(obs, reward, terminated, truncated, info)`。动作 `0` 表示松开、`1` 表示按住。环境只改变模拟按键状态，没有系统键鼠、游戏进程或屏幕操作。

先使用精简的 `geometry-v1` 观测接口：每帧 18 个值，保留 16 帧，加 6 个配置值，共 **294 维**。每帧包括鱼与绿条中心、绿条高度、两者的历史差分速度、进度与进度变化、最后按键状态、切换间隔、时间间隔、有效性标记、画面年龄和剩余时间。所有单位、归一化和字段顺序见 `fishing_sim/observations.py`。

速度来自已发生的位置差分，输入没有引擎真实速度、鱼目标位置、鱼种 ID、难度、运动类别、随机种子或未来轨迹。鱼种信息只放入独立诊断输出及报告。Actor 和 Critic 使用相同观测。后续接图像检测时应向 `ObservationEncoder` 提供同一坐标系的测量，并扩展缺失数据、置信度和延迟处理；当前接口中的有效性固定为真。

PPO 的 Actor/Critic 各为 `[128,128]`、Tanh。参数：16 个向量环境、每环境 256 步 rollout、batch 512、每轮 5 个 epoch、学习率 `3e-4`、gamma `1`、GAE `0.95`、clip `0.2`、熵系数 `0.01`、target KL `0.02`。数值网络使用 CPU；不需要 GPU 依赖。

奖励是捕获 `+1` / 失败或超时 `-1`，加进度势能差、小时间惩罚和小按键切换惩罚。势能为 `0.5*progress`，终止态势能为零；先只优化成功捕获，完美率仅作观测指标。

## 命令行

在本目录打开 PowerShell：

```powershell
# 重建依赖（优先 uv；PyTorch 使用官方 CPU 包）
.\setup.ps1

# 可选：下载固定版本的数值参考，进行源代码对照
.\fetch_reference.ps1
.\.venv\Scripts\python.exe -m fishing_sim.validate

# 新训练，自动创建独立实验目录
.\.venv\Scripts\python.exe -m fishing_sim.train --steps 2000000 --seed 42

# 续训权重及优化器；重新开始环境轨迹，不是逐位恢复中断轨迹
.\.venv\Scripts\python.exe -m fishing_sim.train --steps 2000000 --resume runs/20260927_ppo_v1/latest.zip --no-curriculum

# 对独立种子进行评估；避免反复使用同一批保留测试来调参
.\.venv\Scripts\python.exe -m fishing_sim.evaluate --controller ppo --model runs/20260927_ppo_v1/best.zip --episodes 1000 --seed 2000000 --output runs/ppo_test.json
.\.venv\Scripts\python.exe -m fishing_sim.evaluate --controller mpc --workers 4 --episodes 1000 --seed 2000000 --output runs/mpc_test.json

# 单独启动可视化服务
.\.venv\Scripts\python.exe -m fishing_sim.serve --port 8767
```

命令行训练可以 Ctrl+C 保存停止。后台训练也可在对应实验目录创建名为 `STOP` 的文件；程序在检查点处保存退出。关闭浏览器只关闭页面，训练和本地服务继续运行。新建与续训都会创建新目录，不覆盖已有实验。

## 来源、校验和限制

运动公式以 Pufferdle 固定版本为参照，来源与差异记录在 `SOURCES.md`。核对覆盖五类鱼 × 五种渔具 × 400 帧，共 10,000 个数值状态。另有随机种子重现、Gymnasium/SB3 接口、按键方向、边界、终止奖励和观测隔离检查。

首轮依然是模拟器实验：没有视觉噪声或输入延迟，没有实际游戏校准；一种训练种子也不足以证明算法稳定优于所有基线。高难度鱼、低钓鱼等级需要继续改进，模拟成功率不能直接当作真实游戏成功率。
