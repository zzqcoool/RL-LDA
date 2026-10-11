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

RL_ORIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'orig_stability_linear_nongaussian')
RL_IMPR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'stability_linear_nongaussian')
DEFAULT_ORIG = os.path.join(RL_ORIG, 'rl_traj_seed1.csv')
DEFAULT_IMPR = os.path.join(RL_IMPR, 'rl_traj_seed1.csv')

NAME_ORIG = 'RL-BIC2'
NAME_IMPR = 'RL-LDA'
C_ORIG = 'tab:blue'
C_IMPR = 'tab:red'
TRACK_COLS = ['reward_batch', 'max_reward_batch', 'max_reward']
CONV_EPOCH = 10000
MIN_POINTS = 20

def load_tracks(files):
    
    dfs = [pd.read_csv(f) for f in files]
    n_pts = [len(df) for df in dfs]
    if len(set(n_pts)) > 1:
        raise ValueError(
            f'轨迹采样点数不一致：{n_pts}\n'
            f'可能混入了快速验证的短轨迹（如 200 轮残留）。'
            f'请用自动模式（无参数直接 Run）自动过滤，'
            f'或改用完整轮数的 seed 范围。')
    epoch = dfs[0]['epoch'].to_numpy()
    mean, std = {}, {}
    for col in TRACK_COLS + ['lambda1', 'lambda2']:
        M = np.stack([df[col].to_numpy() for df in dfs])  # (n_seed, n_point)
        mean[col] = M.mean(axis=0)
        std[col] = M.std(axis=0, ddof=1) if len(dfs) > 1 else np.zeros_like(M[0])
    return epoch, mean, std, len(dfs)

def plot_reward_band(ax, epoch, mean, std, color, label, legend_with_suffix=True):
    
    for col, lw, ls, suffix in [('reward_batch', 1.8, '-', 'avg reward / batch'),
                                ('max_reward_batch', 1.4, '--', 'max reward / batch'),
                                ('max_reward', 1.2, ':', 'max reward')]:
        v = -mean[col]
        ax.plot(epoch, v, color=color, lw=lw, ls=ls,
                label=f'{label} {suffix}' if legend_with_suffix else label)
        if std[col].any():
            ax.fill_between(epoch, v - std[col], v + std[col], color=color, alpha=0.15, lw=0)

def plot_avg_band(ax, epoch, mean, std, color, label):
    
    v = -mean['reward_batch']
    ax.plot(epoch, v, color=color, lw=2.2, label=label)
    if std['reward_batch'].any():
        ax.fill_between(epoch, v - std['reward_batch'], v + std['reward_batch'],
                        color=color, alpha=0.18, lw=0)

def conv_stats(files):
    
    import re
    rows = []
    skipped = 0
    for f in files:
        df = pd.read_csv(f)
        m = df['epoch'] >= CONV_EPOCH
        if int(m.sum()) == 0:
            skipped += 1
            continue
        seg = df.loc[m, 'reward_batch']
        mt = re.search(r'seed(\d+)', os.path.basename(f))
        seed = int(mt.group(1)) if mt else -1
        rows.append({'seed': seed,
                     'seg_mean': float(seg.mean()),
                     'seg_std': float(seg.std(ddof=1)) if len(seg) > 1 else 0.0})
    if not rows:
        return {'per_seed': [], 'mean_of_means': float('nan'),
                'sd_of_means': float('nan'), 'mean_of_stds': float('nan'),
                'skipped': skipped}
    seg_means = np.array([r['seg_mean'] for r in rows])
    seg_stds = np.array([r['seg_std'] for r in rows])
    return {'per_seed': rows,
            'mean_of_means': float(seg_means.mean()),
            'sd_of_means': float(seg_means.std(ddof=1)) if len(rows) > 1 else 0.0,
            'mean_of_stds': float(seg_stds.mean()),
            'skipped': skipped}

def auto_seeds(dir_o, dir_m, min_points=MIN_POINTS):
    
    def scan(d):
        found = {}
        for f in glob.glob(os.path.join(d, 'rl_traj_seed*.csv')):
            mt = re.search(r'seed(\d+)', os.path.basename(f))
            if not mt:
                continue
            try:
                with open(f, encoding='utf-8-sig') as fh:
                    n_points = sum(1 for _ in fh) - 1
            except Exception:
                continue
            if n_points >= min_points:
                found[int(mt.group(1))] = f
        return found

    fo, fm = scan(dir_o), scan(dir_m)
    common = sorted(set(fo) & set(fm))
    return common

def main():
    ap = argparse.ArgumentParser(description='RL-BIC2 vs RL-LDA 训练轨迹对比图')
    ap.add_argument('--original', default=None, help='单文件模式：原版轨迹 CSV（默认用多 seed 聚合）')
    ap.add_argument('--improved', default=None, help='单文件模式：改进版轨迹 CSV')
    ap.add_argument('--seed_start', type=int, default=None, help='多 seed 聚合：起始 seed（含）')
    ap.add_argument('--seed_end', type=int, default=None, help='多 seed 聚合：结束 seed（含）')
    ap.add_argument('--dir_orig', default=RL_ORIG, help='原版轨迹所在目录（多 seed 模式）')
    ap.add_argument('--dir_impr', default=RL_IMPR, help='改进版轨迹所在目录（多 seed 模式）')
    ap.add_argument('--out', default='rl_traj_compare.png', help='图1（六曲线带）路径')
    ap.add_argument('--out_avg', default='rl_traj_avg_reward.png', help='图2（两条 avg 曲线带）路径')
    ap.add_argument('--with_lambda', action='store_true',
                    help='同时画出 (a) penalty weights 双子图')
    args = ap.parse_args()

    if args.original and args.improved:
        files_o, files_m = [args.original], [args.improved]
    elif args.seed_start is not None and args.seed_end is not None:
        seeds = list(range(args.seed_start, args.seed_end + 1))
        files_o = [os.path.join(args.dir_orig, f'rl_traj_seed{s}.csv') for s in seeds]
        files_m = [os.path.join(args.dir_impr, f'rl_traj_seed{s}.csv') for s in seeds]
    else:

        seeds = auto_seeds(args.dir_orig, args.dir_impr)
        if not seeds:
            raise FileNotFoundError(
                '未找到两边都存在的完整轨迹（rl_traj_seed*.csv，采样点数≥{}）。\n'
                '请先用原版 demo_stability.py 和 改进版 demo_stability.py 训练至少一个相同的 seed。'
                .format(MIN_POINTS))
        print(f'[AUTO] 自动聚合 seed：{seeds[0]}..{seeds[-1]}（共 {len(seeds)} 个）')
        files_o = [os.path.join(args.dir_orig, f'rl_traj_seed{s}.csv') for s in seeds]
        files_m = [os.path.join(args.dir_impr, f'rl_traj_seed{s}.csv') for s in seeds]

    for f in files_o + files_m:
        if not os.path.exists(f):
            raise FileNotFoundError(f'轨迹文件不存在：{f}\n请先用原版/改进版跑对应 seed 训练。')

    ep_o, mo, so, no = load_tracks(files_o)
    ep_m, mm, sm, nm = load_tracks(files_m)
    n_seed = max(no, nm)
    print(f'[DATA] 原版 {no} 个 seed，改进版 {nm} 个 seed，每轨迹 {len(ep_o)} 个采样点')

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_reward_band(ax, ep_o, mo, so, C_ORIG, NAME_ORIG)
    plot_reward_band(ax, ep_m, mm, sm, C_IMPR, NAME_IMPR)
    ax.set_xlabel('iteration')
    ax.set_ylabel('negative reward')
    ax.set_title(f'Negative reward: {NAME_ORIG} vs {NAME_IMPR}'
                 + (f' (mean±std, {n_seed} seeds)' if n_seed > 1 else ''))
    # ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.savefig(os.path.splitext(args.out)[0] + '.svg', dpi=200)
    print('[SAVE]', os.path.abspath(args.out))

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    plot_avg_band(ax2, ep_o, mo, so, C_ORIG, NAME_ORIG)
    plot_avg_band(ax2, ep_m, mm, sm, C_IMPR, NAME_IMPR)
    ax2.set_xlabel('iteration')
    ax2.set_ylabel('negative reward')
    # ax2.set_title(f'Average reward per batch: {NAME_ORIG} vs {NAME_IMPR}'
    #               )
    # ax2.grid(alpha=0.3)
    ax2.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.savefig(args.out_avg, dpi=200)
    plt.savefig(os.path.splitext(args.out_avg)[0] + '.svg', dpi=200)
    print('[SAVE]', os.path.abspath(args.out_avg))

    so_ = conv_stats(files_o)
    sm_ = conv_stats(files_m)
    if so_ and sm_ and np.isfinite(so_['mean_of_means']) and np.isfinite(sm_['mean_of_means']):
        print('\n' + '=' * 62)
        print(f'收敛段统计（epoch >= {CONV_EPOCH}；负奖励，数值越小越稳定）')
        print('-' * 62)
        print(f'{"method":<10}{"conv mean":<16}{"time-wise std":<16}{"seed-wise std"}')
        print(f'{NAME_ORIG:<10}{so_["mean_of_means"]:<16.4f}{so_["mean_of_stds"]:<16.4f}'
              f'{so_["sd_of_means"]:.4f}')
        print(f'{NAME_IMPR:<10}{sm_["mean_of_means"]:<16.4f}{sm_["mean_of_stds"]:<16.4f}'
              f'{sm_["sd_of_means"]:.4f}')
        print('-' * 62)
        print('注：两方法无环惩罚量级不同，收敛段均值绝对值不可直接比较；')
        print('稳定性以离散度为准：time-wise std（reward 随迭代的波动）越小 = 训练越平稳，')
        print('seed-wise std（不同种子结果的离散）越小 = 各 seed 结果越一致。')
    else:
        print('\n[STATS] 收敛段统计不可用：两侧轨迹均未达到收敛段阈值'
              '（epoch >= {}，至少 1 个采样点）。'.format(CONV_EPOCH))

if __name__ == '__main__':
    main()
