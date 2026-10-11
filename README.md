# RL-LDA

基于强化学习的因果结构发现方法（RL-LDA）。本项目在 ICLR 2020《Causal Discovery with
Reinforcement Learning》(RL-BIC / RL-BIC2, arXiv:1906.04477) 的基础上进行改进，在奖励
函数中引入 DAGMA 的无环约束，在保持 RL 方法灵活得分函数（可处理非光滑得分）优势的同时，
提升训练过程的数值稳定性与结构学习的收敛质量。

## 来源与修改声明

本项目基于 ICLR 2020《Causal Discovery with Reinforcement Learning》(RL-BIC / RL-BIC2, arXiv:1906.04477) 及其官方开源实现（Huawei Noah's Ark Lab, https://github.com/huawei-noah/trustworthyAI）修改而来。在保留 RL 灵活得分函数框架的基础上，本项目的主要改进为：在奖励函数中引入 DAGMA（ICML 2022）的 logdet 无环约束，以提升训练的数值稳定性与结构学习的收敛质量。原始代码依据 Apache License 2.0 授权，本项目的修改与衍生部分同样依据 Apache License 2.0 发布。

## 环境依赖

- Python 3.8（开发环境）
- 依赖清单见 `requirements.txt`：`torch`、`numpy`、`pandas`、`scipy`、`scikit-learn`、
  `matplotlib`、`tqdm`、`openpyxl`
- 可选：`rpy2` + R 包 `CAM`（仅 `demo/demo_sachs.py --prune_mode cam_r`
  真 CAM 剪枝需要；无 rpy2 时自动回退到内置近似剪枝）

安装：

```bash
pip install -r requirements.txt
```

## 目录结构

```
RL-LDA/
├── rl_lda/       核心方法代码（RL 算法、BIC 奖励、DAGMA 约束、评估、数据生成）
├── demo/           快速演示脚本（demo_RL-LDA 线性/二次 SEM（含 ER/SF 备选） / demo_gp GP / demo_sachs Sachs 真实数据 / demo_stability 数值稳定性验证）
├── scripts/        分析与绘图脚本（DAG 对比图、训练轨迹、λ 系数轨迹、CAM 剪枝）
└── data/           实验数据（GP 合成数据、Sachs 数据）
```

## 快速开始（demo）

```bash
# demo_RL-LDA：线性 SEM 数据（示例 12 节点，节点数 d 与噪声类型 gaussian/lingam 均可由头部参数区修改；
# 文件内含二次非线性 SEM 与 ER/SF 随机 DAG 备选块，取消注释即可切换数据模型）
python demo/demo_RL-LDA.py

# demo_gp：GP 非线性数据（读取 data/data_p10_e40_n1000_GP 下的 data{n}.npy / DAG{n}.npy；
# 参数区脚本，修改头部 seed 范围后直接运行）
python demo/demo_gp.py

# demo_sachs：Sachs 真实数据（853 观测样本、11 变量、17 边黄金标准；argparse 脚本，支持批量 seed 与剪枝）
python demo/demo_sachs.py --seed_start 1 --seed_end 10 --nb_epoch 20000

# demo_stability：数值稳定性验证（支持 LR/QR/GPR 三种回归，自动记录奖励轨迹 rl_traj_seed{n}.csv，
# 供 plot_traj.py / plot_lambda.py 绘制训练稳定性图）
python demo/demo_stability.py --reg LR --seed_start 1 --seed_end 10     # 线性 SEM
python demo/demo_stability.py --reg QR --seed_start 1 --seed_end 10     # 二次非线性 SEM
python demo/demo_stability.py --reg GPR --seed_start 1 --seed_end 10    # 读取外部 GP 数据
```

`demo/demo_stability.py` 参数区可配置：`SEED_START / SEED_END / NB_EPOCH / D / N_SAMPLES /
NOISE / REG_TYPE / PROB / DATA_DIR`。`REG_TYPE='GPR'` 时读取 `data/data_p10_e40_n1000_GP`
中的数据文件（异方差噪声，自动使用 `BIC_different_var` 得分与 GPR 回归器）。

## 论文实验复现

```bash
# Sachs 真实数据实验（853 观测样本、11 变量、17 边黄金标准真图；官方超参 nb_epoch=20000）
python demo/demo_sachs.py --seed_start 1 --seed_end 10 --nb_epoch 20000
```

## 结果与绘图

运行各 demo 后，指标 Excel、DAG 对比图与训练轨迹会自动输出到 `results/` 目录。该目录为运行产物，未随仓库上传，首次运行 demo 时自动创建。

- 训练轨迹：`results/stability_*/rl_traj_seed{n}.csv`（各数据模型自动生成对应子目录，含 epoch、λ、奖励）
- DAG 对比图：`results/figures/`（各数据模型的估计图 vs 真图对比，同时输出 PNG 与 SVG）
- 指标 Excel：`results/` 根目录下（如 `linear_results.xlsx`、`gp_results.xlsx`）
- 轨迹聚合绘图（需先运行对应 demo 生成轨迹 CSV）：

```bash
python scripts/plot_traj.py --seed_start 1 --seed_end 10 --dir results/stability_linear_gaussian
python scripts/plot_lambda.py --dir results/stability_linear_gaussian
```

（`plot_dag.py` 用于单 seed 的估计图 vs 真图对比，已内嵌于各 demo 自动调用。）

## 数据说明

- `data/data_p10_e40_n1000_GP/`：10 节点、40 期望边、1000 样本的 GP 非线性合成数据
  （`data{n}.npy` 数据、`DAG{n}.npy` 真图，数据源自 GraN-DAG 仓库：
  https://github.com/kurowasan/GraN-DAG/tree/master/data ）
- `data/sachs/`：Sachs et al. (2005) 蛋白信号网络观测数据（853×11），真图 17 边黄金标准

## 致谢

- 基础框架与数据生成参考华为诺亚方舟实验室开源 RL（
  https://github.com/huawei-noah/trustworthyAI ）与 ICLR 2020《Causal Discovery with
  Reinforcement Learning》
- 无环约束参考 DAGMA（ICML 2022）
