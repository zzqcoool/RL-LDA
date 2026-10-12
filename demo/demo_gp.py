import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rl_lda'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

import os
import numpy as np
import pandas as pd
from rl_lda.rl import RL_LDA
from scripts.plot_dag import GraphDAG
from rl_lda.evaluation import MetricsDAG
from sklearn.preprocessing import StandardScaler
import torch

data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'data_p10_e40_n1000_GP')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
TRAJ_DIR = os.path.join(RESULTS_DIR, 'stability_gp')
excel_filename = os.path.join(RESULTS_DIR, 'gp_results.xlsx')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TRAJ_DIR, exist_ok=True)
results = pd.DataFrame(columns=['seed', 'fdr', 'tpr', 'shd'])

torch.set_default_dtype(torch.float64)

for seed in range(1, 20):
    data_dir = data_root
    X = np.load(os.path.join(data_dir, f'data{seed}.npy'))
    true_dag = np.load(os.path.join(data_dir, f'DAG{seed}.npy'))

    if not (np.allclose(np.mean(X, axis=0), 0, atol=0.1) and
            np.allclose(np.std(X, axis=0), 1, atol=0.1)):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    rl = RL_LDA(nb_epoch=20000, seed=seed, reg_type='GPR', lambda_flag_default=True, input_dimension=128, traj_dir=TRAJ_DIR)
    rl.learn(X)

    GraphDAG(rl.causal_matrix, true_dag, show=False, save_name=os.path.join(FIG_DIR, f'gp-{seed}.png'))
    met = MetricsDAG(rl.causal_matrix, true_dag)
    metrics_dict = met.metrics
    fdr = metrics_dict.get('fdr', np.nan)
    tpr = metrics_dict.get('tpr', np.nan)
    shd = metrics_dict.get('shd', np.nan)

    new_row = pd.DataFrame({'seed': [seed], 'fdr': [fdr], 'tpr': [tpr], 'shd': [shd]})

    if os.path.exists(excel_filename):
        existing = pd.read_excel(excel_filename)
        updated = pd.concat([existing, new_row], ignore_index=True)
        updated.to_excel(excel_filename, index=False)
    else:
        new_row.to_excel(excel_filename, index=False)

    results = pd.concat([results, new_row], ignore_index=True)
    print(f"Seed {seed}: FDR={fdr:.4f}, TPR={tpr:.4f}, SHD={shd}")
