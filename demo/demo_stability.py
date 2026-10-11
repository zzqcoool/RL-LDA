# coding=utf-8

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rl_lda'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

torch.set_default_dtype(torch.float64)

from rl_lda.evaluation import MetricsDAG
from rl_lda.rl import RL_LDA
from rl_lda.synthetic_data_all import generate_W, gen_data_given_model, gen_data_given_model_2nd_order
from scripts.plot_dag import GraphDAG
from sklearn.preprocessing import StandardScaler

SEED_START = 1
SEED_END = 10
NB_EPOCH = 20000
D = 10
N_SAMPLES = 1000
NOISE = 'lingam'
REG_TYPE = 'GPR'
PROB = 0.5
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'data_p10_e40_n1000_GP')
EXCEL = (f'demo_stability_results_{REG_TYPE.lower()}.xlsx' if REG_TYPE == 'GPR'
         else f'demo_stability_results_{REG_TYPE.lower()}_{NOISE}.xlsx')

def main():
    ap = argparse.ArgumentParser(description='demo_stability：循环训练 + 奖励轨迹记录')
    ap.add_argument('--seed_start', type=int, default=SEED_START)
    ap.add_argument('--seed_end', type=int, default=SEED_END)
    ap.add_argument('--nb_epoch', type=int, default=NB_EPOCH)
    ap.add_argument('--n_samples', type=int, default=N_SAMPLES)
    ap.add_argument('--noise', choices=['gaussian', 'lingam'], default=NOISE)
    ap.add_argument('--reg', choices=['LR', 'QR', 'GPR'], default=REG_TYPE,
                    help='回归类型：LR=线性SEM；QR=二次非线性SEM（数据自动按二次方式生成）；GPR=读取外部 GP 数据文件')
    ap.add_argument('--data_dir', type=str, default=DATA_DIR, help='GPR 模式的数据文件夹')
    ap.add_argument('--prob', type=float, default=PROB)
    args = ap.parse_args()

    results = pd.DataFrame(columns=['seed', 'fdr', 'tpr', 'shd', 'total_edges', 'time_s'])
    print(f'[MODE] seed={args.seed_start}..{args.seed_end}, d={D}, n={args.n_samples}, '
          f'noise={args.noise}, reg={args.reg}, epoch={args.nb_epoch}')

    for seed in range(args.seed_start, args.seed_end + 1):
        print(f"\nRunning experiment with seed {seed}...")
        t0 = time.time()
        try:
            np.random.seed(seed)
            d = D
            W = generate_W(d=d, prob=args.prob)   # 0.2
            c = np.zeros(d)
            s = np.ones([d])                      # s = np.round(np.random.uniform(low=0.5, high=2, size=[d]), 1)
            if args.reg == 'GPR':
                xs = np.load(os.path.join(args.data_dir, f'data{seed}.npy'))
                b_ = np.load(os.path.join(args.data_dir, f'DAG{seed}.npy'))
                if not (np.allclose(np.mean(xs, axis=0), 0, atol=0.1) and
                        np.allclose(np.std(xs, axis=0), 1, atol=0.1)):
                    xs = StandardScaler().fit_transform(xs)
            elif args.reg == 'QR':
                xs, b_, c_ = gen_data_given_model_2nd_order(
                    W, s, c, n_samples=args.n_samples, noise_type=args.noise, permutate=True)
                xs_norm = np.linalg.norm(xs, axis=1)
                th_idx = min(3000, args.n_samples - 1)
                xs_th = sorted(xs_norm)[th_idx]
                xs = xs[xs_norm < xs_th]
            else:
                xs, b_, c_ = gen_data_given_model(
                    W, s, c, n_samples=args.n_samples, noise_type=args.noise, permutate=True)
            X = xs                                # (5000, 12)
            true_dag = b_
            for i in range(true_dag.shape[0]):
                for j in range(true_dag.shape[1]):
                    if true_dag[i][j] == 0:
                        pass
                    else:
                        true_dag[i][j] = 1
            X = X.astype(np.float64)

            if args.reg == 'GPR':
                rl = RL_LDA(nb_epoch=args.nb_epoch, reg_type='GPR', seed=seed, input_dimension=128)
            else:
                rl = RL_LDA(nb_epoch=args.nb_epoch, reg_type=args.reg, seed=seed)
            rl.learn(X)

            traj = f'rl_traj_seed{seed}.csv'
            print(f"[TRAJ] 奖励轨迹已记录：{os.path.abspath(traj)}（存在={os.path.exists(traj)}）")

            est = rl.causal_matrix if args.reg == 'GPR' else rl.causal_matrix.T
            GraphDAG(est, true_dag, show=False,
                     save_name=f'demo_stability_seed{seed}.png')

            met = MetricsDAG(est, true_dag)
            fdr = met.metrics['fdr']
            tpr = met.metrics['tpr']
            shd = met.metrics['shd']
            total = int(rl.causal_matrix.sum())
            new_row = pd.DataFrame({'seed': [seed], 'fdr': [fdr], 'tpr': [tpr],
                                    'shd': [shd], 'total_edges': [total],
                                    'time_s': [round(time.time() - t0, 1)]})
            results = pd.concat([results, new_row], ignore_index=True)
            print(f"Seed {seed}: FDR={fdr:.4f}, TPR={tpr:.4f}, SHD={shd}, "
                  f"Total Edges={total}, 用时 {time.time()-t0:.0f}s")

        except Exception as e:
            print(f"Error with seed {seed}: {e}")

    if os.path.exists(EXCEL):
        existing = pd.read_excel(EXCEL)
        results = pd.concat([existing, results], ignore_index=True)
    results.to_excel(EXCEL, index=False)
    print(f"\n[XLSX] 结果已保存：{os.path.abspath(EXCEL)}")

if __name__ == '__main__':
    main()
