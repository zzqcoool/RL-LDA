# coding=utf-8

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIR_IMPR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'stability_linear_gaussian')
MIN_POINTS = 20
COLOR = 'tab:red'

def load_lambda(files):
    
    dfs = [pd.read_csv(f) for f in files]
    epoch = dfs[0]['epoch'].to_numpy()
    out = {}
    for col in ['lambda1', 'lambda2']:
        M = np.stack([df[col].to_numpy() for df in dfs])  # (n_seed, n_point)
        out[col + '_mean'] = M.mean(axis=0)
        out[col + '_std'] = M.std(axis=0, ddof=1) if len(dfs) > 1 else np.zeros_like(M[0])
    return epoch, out, len(dfs)

def main():
    ap = argparse.ArgumentParser(description='改进版 λ1/λ2 训练轨迹图（仿论文 Figure 2(a)）')
    ap.add_argument('--dir', default=DIR_IMPR, help='改进版轨迹目录')
    ap.add_argument('--seed_start', type=int, default=None, help='可选：指定 seed 范围')
    ap.add_argument('--seed_end', type=int, default=None)
    ap.add_argument('--out', default=os.path.join(DIR_IMPR, 'lambda_penalty.png'))
    args = ap.parse_args()

    if args.seed_start is not None and args.seed_end is not None:
        files = [os.path.join(args.dir, f'rl_traj_seed{s}.csv')
                 for s in range(args.seed_start, args.seed_end + 1)]
        files = [f for f in files if os.path.exists(f)]
    else:
        files = sorted(glob.glob(os.path.join(args.dir, 'rl_traj_seed*.csv')))
        files = [f for f in files
                 if sum(1 for _ in open(f, encoding='utf-8-sig')) - 1 >= MIN_POINTS]
    if not files:
        raise FileNotFoundError(f'未找到轨迹文件：{args.dir}')

    n_pts = [sum(1 for _ in open(f, encoding='utf-8-sig')) - 1 for f in files]
    if len(set(n_pts)) > 1:
        raise ValueError(f'轨迹采样点数不一致：{n_pts}，请用 --seed_start/--seed_end 指定完整轨迹范围')
    seeds = sorted(int(re.search(r'seed(\d+)', os.path.basename(f)).group(1)) for f in files)
    print(f'[DATA] 改进版 {len(files)} 个 seed：{seeds[0]}..{seeds[-1]}')

    epoch, m, n_seed = load_lambda(files)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    v, s = m['lambda1_mean'], m['lambda1_std']
    ax1.plot(epoch, v, color=COLOR, lw=1.8)
    ax1.set_ylabel(r'$\lambda_1$', fontsize=13)
    # ax1.grid(alpha=0.3)

    v2, s2 = m['lambda2_mean'], m['lambda2_std']
    ax2.plot(epoch, v2, color=COLOR, lw=1.8)
    ax2.set_yscale('log')
    ax2.set_ylabel(r'$\lambda_2$', fontsize=13)
    ax2.set_xlabel('iteration', fontsize=12)
    # ax2.grid(alpha=0.3, which='both')

    # fig.suptitle(f'Penalty weights: RL-LDA',
    #              fontsize=13)
    # fig.text(0.5, 0.01, '(a) penalty weights', ha='center', fontsize=12)
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    plt.savefig(args.out, dpi=200)
    plt.savefig(os.path.splitext(args.out)[0] + '.svg', dpi=200)
    print('[SAVE]', os.path.abspath(args.out))

if __name__ == '__main__':
    main()
