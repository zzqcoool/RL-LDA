# coding=utf-8
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rl_lda'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import torch
import numpy as np
import pandas as pd
import glob

torch.set_default_dtype(torch.float64)  # use float32 when switching to the 30-node block below

from rl_lda.evaluation import MetricsDAG
from rl_lda.rl import RL_LDA
from rl_lda.synthetic_data_all import generate_W, gen_data_given_model, gen_data_given_model_2nd_order, DAG, IIDSimulation
from scripts.plot_dag import GraphDAG

# os.makedirs('lambda_history', exist_ok=True)
excel_filename = 'linear_results.xlsx'
DATA_TAG = 'linear'  # output name tag: linear / quadratic / gp / 30node; change with the data block below
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'figures')
TRAJ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'stability_linear_nongaussian')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TRAJ_DIR, exist_ok=True)
results = pd.DataFrame(columns=['seed', 'fdr', 'tpr', 'shd'])

for seed in range(1, 2):
    print(f"Running experiment with seed {seed}...")

    # ================================================LINEAR MODEL================================================
    np.random.seed(seed)
    d = 12
    W = generate_W(d=d, prob=0.5)  # 0.2
    c = np.zeros(d)
    s = np.ones([d])
    # s = np.round(np.random.uniform(low=0.5, high=2, size=[d]), 1)   # different varicne
    xs, b_, c_ = gen_data_given_model(W, s, c, n_samples=5000, noise_type='lingam', permutate=True)  # 'lingam','gaussian'

    X = xs  # (5000, 12)
    X = X.astype(np.float64)  # match default dtype
    true_dag = b_
    for i in range(true_dag.shape[0]):
        for j in range(true_dag.shape[1]):
            if true_dag[i][j] == 0:
                pass
            else:
                true_dag[i][j] = 1

    # ================================================QUADRATIC MODEL================================================
    # np.random.seed(seed)
    # d = 10
    # W = generate_W(d=d, prob=0.5, low=0.5, high=2.0)
    # c = np.zeros(d)
    # # s = np.round(np.random.uniform(low=0.5, high=2, size=[d]), 1)
    # s = np.ones([d])
    # xs, b_, c_ = gen_data_given_model_2nd_order(W, s, c, n_samples=5000, noise_type='lingam', permutate=True)  # 'lingam','gaussian'
    #
    # xs_norm = np.linalg.norm(xs, axis=1)
    # xs_th = sorted(xs_norm)[3000]
    # xs = xs[xs_norm < xs_th]

    # X = xs
    # true_dag = b_
    # for i in range(true_dag.shape[0]):
    #     for j in range(true_dag.shape[1]):
    #         if true_dag[i][j] == 0:
    #             pass
    #         else:
    #             true_dag[i][j] = 1
    # X = X.astype(np.float64)


    # ================================================ER/SF================================================
    # type = 'ER'  # 'ER' or 'SF'
    # h = 2
    # n_nodes = 12
    # n_edges = h * n_nodes
    # method = 'linear'
    # sem_type = 'gauss'
    # if type == 'ER':
    #     weighted_random_dag = DAG.erdos_renyi(n_nodes=n_nodes, n_edges=n_edges, weight_range=(0.5, 2.0), seed=seed)
    # elif type == 'SF':
    #     weighted_random_dag = DAG.scale_free(n_nodes=n_nodes, n_edges=n_edges, weight_range=(0.5, 2.0), seed=seed)
    # else:
    #     raise ValueError('Just supported `ER` or `SF`.')
    # dataset = IIDSimulation(W=weighted_random_dag, n=2000, method=method, sem_type=sem_type)
    # true_dag, X = dataset.B, dataset.X

    # rl learn
    rl = RL_LDA(nb_epoch=20000, reg_type='LR', seed=seed, record_lambda_history=True, traj_dir=TRAJ_DIR)

    rl.learn(X)

    # ================================================EVALUATION================================================
    # Direction convention: gen_data_given_model family (linear/quadratic) returns a
    # lower-triangular row-weight matrix b_, so the estimate needs a transpose to align;
    # ER/SF (IIDSimulation) and GP data files use standard i->j adjacency (no transpose).
    use_transpose = bool(np.allclose(true_dag, np.tril(true_dag)))
    est = rl.causal_matrix if use_transpose else rl.causal_matrix.T
    GraphDAG(est, true_dag, show=False, save_name=os.path.join(FIG_DIR, f'{DATA_TAG}-{seed}.png'))

    met = MetricsDAG(est, true_dag)
    metrics_dict = met.metrics

    fdr = metrics_dict.get('fdr', np.nan)
    tpr = metrics_dict.get('tpr', np.nan)
    shd = metrics_dict.get('shd', np.nan)

    new_row = pd.DataFrame({'seed': [seed], 'fdr': [fdr], 'tpr': [tpr], 'shd': [shd]})

    if os.path.exists(excel_filename):
        existing_data = pd.read_excel(excel_filename)
        updated_data = pd.concat([existing_data, new_row], ignore_index=True)
        updated_data.to_excel(excel_filename, index=False)
    else:
        new_row.to_excel(excel_filename, index=False)

    results = pd.concat([results, new_row], ignore_index=True)
    print(f"Seed {seed}: FDR = {fdr:.4f}, TPR = {tpr:.4f}, SHD = {shd}")
