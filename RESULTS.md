# 复现结果

## 最终结论

本项目已经在独立仿真器中复现论文的两个核心定量结果，但两者不是同一条端到端训练链：

- Fig. 7A 教师三种子曲线已完整复现。IR 在约 1B 样本处的复现均值为 **97.46%**，
  论文为约 **96.00%**；no-IR 在 999.424M 的复现均值为 **64.75%**，论文为
  **68.23%**。
- 图像学生在 30°、60°、80° 静态滚转窄缝上的成功率为 **99.4%**、**98.4%**、
  **86.9%**，论文 Table S4 unified 策略为 **100%**、**95.7%**、**88.6%**。
- 学生结果由较早的 201M 教师监督。最终稳定 1B 教师到学生的从零蒸馏未达到 Table S4，
  因此不能声称已经复现“最终教师→最终学生”的单一端到端链路。

复现范围严格限定为单个静态矩形滚转窄缝。真实硬件、学习分割、动态窄缝、连续窄缝、
俯仰窄缝、其他几何形状及论文全部基线尚未复现，当前结果不能验证硬件或 sim-to-real 主张。

## 仿真定性结果

已生成同一标准起点（seed 20080）、80° 滚转窄缝的教师和图像学生成功穿越视频。推荐查看
包含简易 RGB 机载视角的版本：

- 教师 RGB 视频：`artifacts/qualitative/teacher_roll80_rgb_success.mp4`
- 教师 RGB 关键帧：`artifacts/qualitative/teacher_roll80_rgb_success_keyframes.png`
- 学生 RGB 视频：`artifacts/qualitative/student_roll80_rgb_success.mp4`
- 学生 RGB 关键帧：`artifacts/qualitative/student_roll80_rgb_success_keyframes.png`

视频同时显示世界坐标轨迹、简易合成 RGB 机载画面、二值 landmark mask 小窗、正视角下
机体包围框与窄缝的姿态对齐，以及速度、滚转角和控制指令。教师在 108 个 60 Hz 控制步后
成功，学生在 105 步后成功；视频以 30 FPS 播放，每帧对应两个控制步。RGB 画面由天空、
地面网格、程序化混凝土墙面和橙色窄缝框组成，只用于展示。学生小窗中的二值 mask 是策略
实际输入；教师不使用视觉输入，其 mask 仅供参考。

这些画面来自本项目依据论文独立实现的轻量级动力学与几何仿真器，不是作者未公开的
Flightmare/Unity 原始场景，也不是照片级渲染。可用以下命令重新寻找成功 episode 并录像：

```bash
python3 scripts/render_policy_rollout.py teacher \
  artifacts/teachers/reproduced_teacher_stable_1b.pt \
  --angle 80 --output artifacts/qualitative/teacher_roll80_rgb_success.mp4
python3 scripts/render_policy_rollout.py student \
  artifacts/students/reproduced_student.pt \
  --angle 80 --output artifacts/qualitative/student_roll80_rgb_success.mp4
```

## Fig. 7A 教师曲线

正式产物：

- 图：`artifacts/training/fig7a_reproduced_final.png`
- 聚合数据：`artifacts/training/fig7a_reproduced_final.csv`
- 误差表：`artifacts/training/fig7a_reproduced_final_metrics.txt`
- 论文参考数据：`reference_data/fig7a_training_curves.csv`

两种条件均使用三个训练种子，每 8.192M 样本进行 4096-episode 确定性标准起点评估。

| 条件 | 公共点数 | RMSE | MAE | 最后公共点复现均值 | 论文均值 |
|---|---:|---:|---:|---:|---:|
| IR | 121 | 4.87 个百分点 | 4.27 个百分点 | 97.46% @ 991.232M | 96.00% |
| no-IR | 122 | 3.78 个百分点 | 2.88 个百分点 | 64.75% @ 999.424M | 68.23% |

关键采样点如下：

| 样本数 | IR 复现 | IR 论文 | no-IR 复现 | no-IR 论文 |
|---:|---:|---:|---:|---:|
| 122.880M | 26.27% | 14.24% | 0.69% | 2.49% |
| 303.104M | 74.44% | 68.22% | 42.46% | 36.74% |
| 499.712M | 87.36% | 82.82% | 60.30% | 60.40% |
| 802.816M | 96.72% | 89.82% | 65.38% | 66.68% |
| 991.232M | 97.46% | 96.00% | 65.96% | 65.81% |

no-IR 的均值与论文接近，但随机种子方差没有复现：999.424M 时三个种子分别为
96.14%、0% 和 98.10%，复现标准差为 45.79 个百分点，论文仅 4.84 个百分点。这个结果
说明均值匹配依赖一个完全失败的种子，必须与均值一起披露，不能只看曲线中心。

IR 原始固定学习率训练也存在峰后回落：seed275 在 827.392M 达到 97.1%，但在
999.424M 降至 80.3%；seed282 在 983.040M 为 95.4%，最后一个评估周期降至 79.9%。
最终协议对所有 IR 种子使用相同的 604.5696M 切换点，将 actor 学习率从 `3e-4` 降到
`1e-4` 并冻结 observation normalizer。三条稳定 run 在约 997.8M 分别为 98.34%、
97.29% 和 97.02%。这是明确的复现稳定性假设，不是论文公开设置。

## 发布教师

主检查点：

- `artifacts/teachers/reproduced_teacher_stable_1b.pt`
- 来源：`runs/rewardfix_ir_stable604_seed274/checkpoints/teacher_final.pt`
- SHA-256：`5f55136d4a7d34708e498d2e37049b79a237c324abc356ca85a1d93f525c0e0b`

正式评估使用随机角度 8192 个 episode、每个固定角度 4096 个 episode：

| 评估设置 | 成功率 | 结果 |
|---|---:|---:|
| 随机角度，确定性均值动作 | 98.3% | 8049/8192，143 次碰撞 |
| 随机角度，随机采样动作 | 98.2% | 8047/8192，145 次碰撞 |
| 固定 0°，确定性 | 100.0% | 4096/4096 |
| 固定 30°，确定性 | 100.0% | 4096/4096 |
| 固定 60°，确定性 | 99.9% | 4090/4096 |
| 固定 80°，确定性 | 95.6% | 3914/4096 |
| 固定 90°，确定性 | 72.8% | 2980/4096 |

原始输出为 `artifacts/training/reproduced_teacher_stable604_final_evaluation.txt`。旧的固定
`3e-4` 检查点 `artifacts/teachers/reproduced_teacher_faithful_806m.pt` 保留用于消融和
可追溯性；其随机角度确定性成功率为 96.3%。

## 发布学生

检查点：

- `artifacts/students/reproduced_student.pt`
- 来源：`runs/distill_high_angle_finetune_lr1e5/checkpoints/student_900.pt`
- SHA-256：`67809b329ea0366d832c462fe59d344838db5ff21619b5302ae6ab25c0c59cef`

每个角度使用 4096 个标准起点 episode。论文 Table S4 同时报告 unified 多专家学生和
separate 专家学生；当前实现只训练滚转任务，训练协议更接近 separate，但数值也列出
unified 作为最接近的统一策略对照。

| 滚转角 | 复现：干净 mask | 复现：随机化 mask | 论文 unified | 论文 separate |
|---:|---:|---:|---:|---:|
| 30° | 99.4% | 98.3% | 100.0% | 100.0% |
| 60° | 98.4% | 98.5% | 95.7% | 97.5% |
| 80° | 86.9% | 87.0% | 88.6% | 92.3% |

论文不报告 0° 和 90°，所以这两个角度不进入主表。当前课程使边界角度出现灾难性遗忘：
干净 mask 下 0°/90° 均约 26.5%。第 900 轮优于最终轮；第 1000 轮在 30°、60°、80°
上为 98.4%、97.0%、77.9%，说明不能仅按监督损失或最后一轮选模型。

原始输出：

- `artifacts/training/reproduced_student_900_fixed_angles.txt`
- `artifacts/training/reproduced_student_900_randomized_masks.txt`
- `artifacts/training/reproduced_student_900_edge_angles.txt`

成功学生由 `artifacts/teachers/reproduced_teacher_201m.pt` 监督：先全角度训练 300 轮，
再在 60-90° 上以 `1e-5` 训练到 1000 轮。它不是最终发布教师的蒸馏结果。

### 最终教师蒸馏复核

对 806M 教师和稳定 1B 教师分别进行了相同的两阶段从零蒸馏，并测试了 `1e-5`、`2e-5`
和 `5e-5` 高角度学习率。结果均未达到 Table S4：

- 806M 教师从零蒸馏第 700 轮约为 31.2%/66.4%/42.6%（256 episode 筛选）；
- 稳定 1B 教师从零蒸馏第 700 轮为 4.4%/49.6%/35.1%（每角 1024 episode）；
- 从发布学生热启动并改用稳定教师监督，第 1000 轮降至 7.8%/37.1%/38.8%。

这些负结果表明不同高成功率教师的动作分布和可蒸馏性差异很大。当前项目复现了教师曲线
和学生表格的数值现象，但没有恢复作者未公开的、同时兼顾教师性能与学生可蒸馏性的训练链。

## 关键复现条件

1. 动作幅值和动作变化惩罚作用于归一化策略动作；直接惩罚物理推力和角速度会压制激进
   滚转机动。
2. 高斯动作分布初始 `log_std=-2.0`，IR 下限为 `-3.5`，no-IR 下限为 `-3.175`。
   论文未公开动作方差，这是最关键的缺失超参数。
3. IR 概率为 0.5，轨迹文件为 `artifacts/ir/optimized_single_gap_roll.pt`。
4. 使用正文描述的 4 m/s 以上超速惩罚，而不是补充材料公式按字面解释得到的低速奖励。
5. 硬 KL 超限时在优化器步之前拒绝更新，并恢复 actor 与 Adam 状态。
6. IR 在 604.5696M 统一切换到 actor LR `1e-4` 并冻结 normalizer。
7. 学生续训会覆盖 Adam 检查点恢复的旧学习率，并在 episode 边界重置循环隐藏状态。

若不限制动作方差，`log_std` 会下降到约 `-4`，随后频繁触发 KL 回滚；actor 冻结后
normalizer 继续漂移还会造成进一步性能回落。

## 信息化重置验证

额外构建的多起点 IR 敏感性库含 7 个角度锚点、每个角度 9 个初始偏移量、每条轨迹 251
个状态。63 条锚点路径和 0-90° 整数角插值得到的 819 条路径全部通过精确长方体碰撞与平面
穿越检查：锚点和插值路径均为零碰撞、全部成功穿越。产物为
`artifacts/ir/optimized_single_gap_roll_grid.pt`；它不是发布教师所用的 IR 文件。

## 无法精确还原的内容

论文与 Zenodo 存档没有提供原始仿真器、训练代码、检查点、IR 状态库、完整 reset 边界、
动作方差、全部 PPO 设置、拟合执行器参数、分割数据集或部署代码。Zenodo 主要包含绘图数据
与 ROS bag。因此当前实现是依据公开文字、图表、参考曲线和相关开源动力学重新构建的独立
复现，不是作者私有系统的源码恢复。

## 验证状态

单元测试结果为 `15 passed`，覆盖碰撞几何、四元数积分、IR 路径终止、奖励解释、归一化
动作惩罚、动作映射、学生网络尺寸、循环隐藏状态重置和二值渲染。完整命令：

```bash
python3 -m pytest -q
```
