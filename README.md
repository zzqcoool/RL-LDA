# RL-LDA

Reinforcement-learning-based causal structure discovery with an improved acyclicity
constraint. This project builds upon the ICLR 2020 paper *Causal Discovery with
Reinforcement Learning* (RL-BIC / RL-BIC2, arXiv:1906.04477). Our key improvement is the
introduction of the DAGMA log-determinant acyclicity constraint into the reward function,
which preserves the flexibility of RL-based score functions (including non-smooth scores)
while improving the numerical stability of training and the convergence quality of the
learned structure.

## Source and Modifications

This project is modified from the ICLR 2020 paper *Causal Discovery with Reinforcement
Learning* (RL-BIC / RL-BIC2, arXiv:1906.04477) and its official open-source implementation
(Huawei Noah's Ark Lab, https://github.com/huawei-noah/trustworthyAI). While keeping the
flexible RL score-function framework unchanged, the main contribution of this project is
incorporating the logdet acyclicity constraint of DAGMA (ICML 2022) into the reward
function to improve training stability and structural convergence. The original code is
released under the Apache License 2.0; the modifications and derived parts of this project
are likewise distributed under the Apache License 2.0.

## Requirements

- Python 3.8 (development environment)
- Dependencies are listed in `requirements.txt`: `torch`, `numpy`, `pandas`, `scipy`,
  `scikit-learn`, `matplotlib`, `tqdm`, `openpyxl`
- Optional: `rpy2` + the R package `CAM` (only needed for the true CAM pruning in
  `demo/demo_sachs.py --prune_mode cam_r`; without `rpy2` the code automatically falls
  back to the built-in approximate pruning)

Install:

```bash
pip install -r requirements.txt
```

## Repository Structure

```
RL-LDA/
├── rl_lda/       Core method code (RL algorithm, BIC reward, DAGMA constraint, evaluation, data generation)
├── demo/         Quick demo scripts (demo_RL-LDA linear/quadratic SEM (with ER/SF alternatives) / demo_gp GP / demo_sachs Sachs real data / demo_stability numerical-stability verification)
├── scripts/      Analysis and plotting scripts (DAG comparison plots, training trajectories, lambda-coefficient trajectories, CAM pruning)
└── data/         Experimental data (GP synthetic data, Sachs data)
```

## Quick Start (Demos)

```bash
# demo_RL-LDA: linear SEM data (example 12 nodes; the node count d and the noise type
# gaussian/lingam can be changed in the parameter block at the top of the file;
# the file also contains quadratic nonlinear SEM and ER/SF random-DAG alternatives,
# uncomment the corresponding block to switch data models)
python demo/demo_RL-LDA.py

# demo_gp: GP nonlinear data (reads data{n}.npy / DAG{n}.npy under data/data_p10_e40_n1000_GP;
# parameter-block script; edit the seed range at the top and run directly)
python demo/demo_gp.py

# demo_sachs: Sachs real data (853 observational samples, 11 variables, 17-edge gold
# standard; argparse-based script, supports batched seeds and pruning)
python demo/demo_sachs.py --seed_start 1 --seed_end 10 --nb_epoch 20000

# demo_stability: numerical-stability verification (supports LR/QR/GPR regressors,
# automatically records reward trajectories rl_traj_seed{n}.csv, which are used by
# plot_traj.py / plot_lambda.py to produce training-stability figures)
python demo/demo_stability.py --reg LR --seed_start 1 --seed_end 10     # linear SEM
python demo/demo_stability.py --reg QR --seed_start 1 --seed_end 10     # quadratic nonlinear SEM
python demo/demo_stability.py --reg GPR --seed_start 1 --seed_end 10    # external GP data
```

The parameter block of `demo/demo_stability.py` exposes: `SEED_START / SEED_END /
NB_EPOCH / D / N_SAMPLES / NOISE / REG_TYPE / PROB / DATA_DIR`. With `REG_TYPE='GPR'`,
the data files under `data/data_p10_e40_n1000_GP` are loaded (heteroscedastic noise,
and the `BIC_different_var` score with the GPR regressor is used automatically).

## Reproducing the Paper Experiments

```bash
# Sachs real-data experiment (853 observational samples, 11 variables, 17-edge gold
# standard; official hyper-parameter nb_epoch=20000)
python demo/demo_sachs.py --seed_start 1 --seed_end 10 --nb_epoch 20000
```

## Results and Plotting

After running any demo, the metric Excel files, DAG comparison figures and training
trajectories are written automatically to the `results/` directory. This directory is a
runtime output and is not tracked by the repository; it is created automatically on the
first run.

- Training trajectories: `results/stability_*/rl_traj_seed{n}.csv` (a corresponding
  sub-directory is created for each data model; columns include epoch, lambda, reward)
- DAG comparison figures: `results/figures/` (estimated graph vs. ground-truth graph for
  each data model, exported as both PNG and SVG)
- Metric Excel files: directly under `results/` (e.g. `linear_results.xlsx`,
  `gp_results.xlsx`)
- Aggregated trajectory plotting (run the corresponding demo first to generate the
  trajectory CSVs):

```bash
python scripts/plot_traj.py --seed_start 1 --seed_end 10 --dir results/stability_linear_gaussian
python scripts/plot_lambda.py --dir results/stability_linear_gaussian
```

(`plot_dag.py` draws the estimated-vs-true graph comparison for a single seed; it is
already invoked automatically inside the demos.)

## Data

- `data/data_p10_e40_n1000_GP/`: GP nonlinear synthetic data with 10 nodes, 40 expected
  edges and 1000 samples (`data{n}.npy` observations, `DAG{n}.npy` ground-truth graphs;
  data from the GraN-DAG repository:
  https://github.com/kurowasan/GraN-DAG/tree/master/data )
- `data/sachs/`: Sachs et al. (2005) protein-signaling network observational data
  (853x11), with the 17-edge gold-standard ground truth

## Acknowledgement

- The base framework and data generation follow the open-source RL implementation of
  Huawei Noah's Ark Lab (https://github.com/huawei-noah/trustworthyAI) and the ICLR 2020
  paper *Causal Discovery with Reinforcement Learning*
- The acyclicity constraint follows DAGMA (ICML 2022)
