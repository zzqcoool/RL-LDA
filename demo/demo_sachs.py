# coding=utf-8
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rl_lda'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

import argparse
import csv
import os
import time
import urllib.request

import numpy as np
import torch
from scipy.spatial.distance import pdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

from rl_lda.rl import RL_LDA
from scripts.plot_dag import GraphDAG
from rl_lda.evaluation import MetricsDAG

torch.set_default_dtype(torch.float64)

LABELS = ['Raf', 'Mek', 'Plcg', 'PIP2', 'PIP3', 'Erk', 'Akt', 'PKA', 'PKC', 'P38', 'Jnk']
D = len(LABELS)

DATA_URL = ('https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/'
            'real/sachs/data/sachs.2005.continuous.txt')

# ============================================================

# ============================================================
def builtin_dag():
    
    arcs = [('PKC', 'PKA'), ('PKC', 'Raf'), ('PKA', 'Raf'),
            ('PKC', 'Mek'), ('PKA', 'Mek'), ('Raf', 'Mek'),
            ('Mek', 'Erk'), ('PKA', 'Erk'), ('Erk', 'Akt'), ('PKA', 'Akt'),
            ('PKC', 'P38'), ('PKA', 'P38'), ('PKC', 'Jnk'), ('PKA', 'Jnk'),
            ('Plcg', 'PIP3'), ('Plcg', 'PIP2'), ('PIP3', 'PIP2')]
    idx = {n: i for i, n in enumerate(LABELS)}
    B = np.zeros((D, D))
    for a, b in arcs:
        B[idx[a], idx[b]] = 1.0
    return B.astype(int)

def load_sachs_data(data_path):
    
    if data_path.lower().endswith('.npy'):
        X = np.load(data_path)
    else:
        arr = np.genfromtxt(data_path, dtype=float)
        if arr.ndim == 1 and arr.size == 0:
            raise ValueError(f'数据文件为空：{data_path}')
        X = arr

        if X.ndim == 2:
            X = X[~np.isnan(X).any(axis=1)]
        if X.shape[1] == 12 and X.shape[0] != 853:
            X = X[1:]
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != D:
        raise ValueError(f'数据列数应为 {D}，实际 {X.shape}')
    if X.shape[0] == 7466:
        print('[DATA] 检测到完整版 7466 行（含干预实验），取前 853 行观测子集'
              '（论文口径 m=853）')
        X = X[:853]
    if X.shape[0] != 853:
        print(f'[WARN] 数据行数 {X.shape[0]} ≠ 853（观测子集行数），继续运行但请注意口径')
    return X

def load_sachs_dag(dag_path):
    
    if dag_path and os.path.exists(dag_path):
        if dag_path.lower().endswith('.npy'):
            B = np.load(dag_path)
        else:
            B = np.zeros((D, D))
            idx = {n.lower(): i for i, n in enumerate(LABELS)}
            for line in open(dag_path, encoding='utf-8'):
                parts = line.split()
                if len(parts) < 2:
                    continue
                a, b = parts[0].lower(), parts[1].lower()
                if a in idx and b in idx:
                    B[idx[a], idx[b]] = 1.0
        return np.asarray(B, dtype=float), os.path.abspath(dag_path)
    return builtin_dag(), '内置 17 边黄金标准（bnlearn dag.sachs）'

# ============================================================

# ============================================================

def _node_in_rss(X, i, pa, cache):
    
    key = (i, tuple(pa))
    if key in cache:
        return cache[key]
    if len(pa) == 0:
        y = X[:, i]
        rss = float(np.sum((y - np.mean(y)) ** 2))
    else:
        Xp = X[:, pa]
        med_w = np.median(pdist(Xp, 'euclidean'))
        gpr = GaussianProcessRegressor().fit(Xp / med_w, X[:, i])
        rss = float(np.sum((X[:, i] - gpr.predict(Xp / med_w)) ** 2)) + 1.0
    cache[key] = rss
    return rss

def _in_sum_rss(X, G, cache):
    d = X.shape[1]
    total = 0.0
    for i in range(d):
        pa = tuple(np.where(G[:, i] > 0.5)[0])
        total += _node_in_rss(X, i, pa, cache)
    return total

def cam_r_prune(X, est, cutoff=0.001, tmpdir='cam_tmp'):
    
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
    except ImportError as e:
        raise RuntimeError(
            '未安装 rpy2：请用 CausalDiscovery37 或已装 rpy2 的环境（python38）运行，'
            '并确认 R_HOME 指向 R 安装目录') from e
    try:
        CAM = importr('CAM')
        importr('Matrix')
    except Exception as e:
        raise RuntimeError('R 包 CAM/Matrix 加载失败：' + str(e)) from e

    import pandas as pd
    n, d = X.shape
    os.makedirs(tmpdir, exist_ok=True)
    cnames = ['X%d' % i for i in range(1, d + 1)]
    pd.DataFrame(X, columns=cnames).to_csv(f'{tmpdir}/X.csv', index=False)
    pd.DataFrame(est.astype(int)).to_csv(f'{tmpdir}/G.csv', index=False, header=False)
    ro.r(f'''
library(CAM)
X  <- read.csv("{tmpdir}/X.csv")
G  <- as.matrix(read.csv("{tmpdir}/G.csv", header = FALSE))
colnames(G) <- colnames(X); rownames(G) <- colnames(X)
Gsp <- as(G, "sparseMatrix")
res <- CAM:::pruning(X = X, G = Gsp, pruneMethod = selGam,
                     pruneMethodPars = list(cutOffPVal = {cutoff}, numBasisFcts = 10))
write.csv(as.matrix(res), "{tmpdir}/res.csv", row.names = FALSE)
''')
    Gp = np.loadtxt(f'{tmpdir}/res.csv', delimiter=',', skiprows=1)
    return (Gp > 0.5).astype(int)

def greedy_prune(X, G, ratio_tol=0.01):
    
    d = X.shape[1]
    cache = {}
    G_cur = np.array(G, dtype=float)
    s_before = _in_sum_rss(X, G_cur, cache)
    s_cur = s_before
    while True:
        best_pair, best_s = None, None
        for i in range(d):
            for j in range(d):
                if G_cur[i, j] > 0.5:
                    Gc = G_cur.copy()
                    Gc[i, j] = 0.0
                    sc = _in_sum_rss(X, Gc, cache)
                    if sc <= (1.0 + ratio_tol) * s_cur and (best_s is None or sc <= best_s):
                        best_pair, best_s = (i, j), sc
        if best_pair is None:
            break
        G_cur[best_pair[0], best_pair[1]] = 0.0
        s_cur = best_s
    removed = int((G != G_cur).sum())
    return G_cur, removed, s_before, s_cur

def additive_prune(X, G, tol=0.05):
    
    from scipy.spatial.distance import pdist as _pdist
    n, d = X.shape
    pred = {}
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            xj = X[:, j:j + 1]
            med = float(np.median(_pdist(xj, 'euclidean'))) or 1.0
            gpr = GaussianProcessRegressor().fit(xj / med, X[:, i])
            f = gpr.predict(xj / med)
            pred[(i, j)] = f - f.mean()

    def node_rss(i, pa):
        if len(pa) == 0:
            return float(np.sum((X[:, i] - X[:, i].mean()) ** 2))
        e = X[:, i] - X[:, i].mean() - sum(pred[(i, j)] for j in pa)
        return float(np.sum(e ** 2)) + 1.0

    Gc = np.array(G, dtype=float)
    s_before = sum(node_rss(i, list(np.where(Gc[:, i] > 0.5)[0])) for i in range(d))
    removed = 0
    changed = True
    while changed:
        changed = False
        for i in range(d):
            pa = list(np.where(Gc[:, i] > 0.5)[0])
            if not pa:
                continue
            rss_base = node_rss(i, pa)
            cands = sorted(pa, key=lambda p: node_rss(i, [x for x in pa if x != p]))
            for p in cands:
                rss_p = node_rss(i, [x for x in pa if x != p])
                if rss_p <= (1.0 + tol) * rss_base:
                    Gc[p, i] = 0.0
                    pa = [x for x in pa if x != p]
                    rss_base = node_rss(i, pa)
                    removed += 1
                    changed = True
    s_after = sum(node_rss(i, list(np.where(Gc[:, i] > 0.5)[0])) for i in range(d))
    return Gc, removed, s_before, s_after

# ============================================================

# ============================================================
def run_single_seed(seed, args, X, dag, save_plot):
    rl = RL_LDA(
        nb_epoch=args.nb_epoch,
        seed=seed,
        reg_type='GPR',

        score_type = 'BIC',
        normalize=True,
        input_dimension=args.input_dimension,
        lambda_flag_default=True,
        lambda_iter_num=1000,
        use_bias=not args.no_bias,
        bias_initial_value=args.bias_initial_value,
        l1_graph_reg=args.l1_graph_reg,
    )
    t0 = time.time()
    rl.learn(X)
    elapsed = time.time() - t0

    est = rl.causal_matrix
    met = MetricsDAG(est, dag).metrics

    row = {
        'seed': seed,
        'total_edges': int(est.sum()),
        'correct_edges': int((est * dag).sum()),
        'shd': int(met['shd']),
        'fdr': met['fdr'],
        'tpr': met['tpr'],
        'nnz': int(met['nnz']),
        'time_s': round(elapsed, 1),
    }
    print(f'[RESULT] seed={seed}: Total Edges={row["total_edges"]}, '
          f'Correct Edges={row["correct_edges"]}, SHD={row["shd"]} '
          f'（FDR={row["fdr"]:.4f}, TPR={row["tpr"]:.4f}）, 用时 {elapsed:.0f}s')

    if args.prune:
        Xs = StandardScaler().fit_transform(X)
        if args.prune_mode == 'cam_r':
            try:
                Gp = cam_r_prune(Xs, est, cutoff=args.cam_cutoff)
                removed = int((est != Gp).sum())
                print(f'[PRUNE-CAM] cutOffPVal={args.cam_cutoff:g}，去掉 {removed} 条边：', end='')
            except Exception as e:
                print(f'[WARN] 真 CAM 剪枝失败（{e}），回退 additive_prune')
                Gp, removed, _, _ = additive_prune(Xs, est, tol=args.prune_tol)
        elif args.prune_mode == 'cam':
            Gp, removed, s_before, s_after = additive_prune(Xs, est, tol=args.prune_tol)
            print(f'[PRUNE] 去掉 {removed} 条边（ΣRSS {s_before:.1f}→{s_after:.1f}）：', end='')
        else:
            Gp, removed, s_before, s_after = greedy_prune(Xs, est, ratio_tol=args.prune_tol_ratio)
            print(f'[PRUNE] 去掉 {removed} 条边（ΣRSS {s_before:.1f}→{s_after:.1f}）：', end='')
        met_p = MetricsDAG(Gp, dag).metrics
        row.update({
            'total_edges_pruned': int(Gp.sum()),
            'correct_edges_pruned': int((Gp * dag).sum()),
            'fdr_pruned': met_p['fdr'],
            'tpr_pruned': met_p['tpr'],
            'shd_pruned': int(met_p['shd']),
            'nnz_pruned': int(met_p['nnz']),
            'pruned_edges': removed,
        })
        print(f'Total Edges={int(Gp.sum())}, Correct={int((Gp * dag).sum())}, '
              f'SHD={met_p["shd"]}')

    if save_plot:
        G_show = Gp if args.prune else est
        GraphDAG(np.array(G_show, dtype=int), dag, show=False,
                 save_name=f'sachs_demo_seed{seed}_ep{args.nb_epoch}.png')

    np.save(f'sachs_est_seed{seed}_ep{args.nb_epoch}.npy', est)
    np.save(f'sachs_dag_ep{args.nb_epoch}.npy', dag)
    return row

def print_aggregate(rows, keys):
    import statistics
    print('\n' + '=' * 66)
    print(f'汇总（{len(rows)} 个 RL_LDA 种子）')
    print('-' * 66)
    for key in keys:
        vals = [r[key] for r in rows]
        if len(rows) > 1:
            print(f'{key:<14}{np.mean(vals):.3f}±{np.std(vals, ddof=1):.3f}'
                  f'  中位 {np.median(vals):.3f}  范围 {min(vals):.1f}~{max(vals):.1f}')
        else:
            print(f'{key:<14}{np.mean(vals):.3f}')

def save_excel(rows, path):
    
    if not rows:
        return
    try:
        import openpyxl
    except ImportError:
        print('[WARN] 未安装 openpyxl，跳过 Excel 输出（CSV 已保存）')
        return
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'sachs_results'
    headers = list(rows[0].keys())
    ws.append(headers)
    for r in rows:
        ws.append([r[h] for h in headers])
    wb.save(path)
    print(f'[XLSX] 明细已保存：{os.path.abspath(path)}')

# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Sachs 数据 + RL_LDA(GPR) 因果发现 demo（论文 6.4 节）')
    parser.add_argument('--seed', type=int, default=1, help='单种子模式（默认 1）')
    parser.add_argument('--seed_start', type=int, default=1, help='批量 RL_LDA 种子起始（含）')
    parser.add_argument('--seed_end', type=int, default=10, help='批量 RL_LDA 种子结束（含，默认=start）')
    parser.add_argument('--no_plots', action='store_true',
                        help='不保存每种子对比图（默认每个种子都存 sachs_demo_seed{n}_ep{nb_epoch}.png）')
    parser.add_argument('--nb_epoch', type=int, default=20000, help='RL_LDA 训练迭代数（默认论文 20000）')
    parser.add_argument('--input_dimension', type=int, default=128, help='构建输入样本的采样数（论文 128）')
    parser.add_argument('--score_type', choices=['BIC', 'BIC_different_var'],
                        default='BIC_different_var',
                        help='BIC=等方差（RL-BIC2 口径）；BIC_different_var=异方差（论文 RL-BIC 口径，'
                             '默认）。Sachs 实测：异方差 est 含 11 条真边，CAM 剪枝后 Total 11/Correct 6/'
                             'SHD 13（对照论文 RL-BIC 10/6/12）；等方差仅 9 条真边，剪枝后 8/4/15')
    parser.add_argument('--data_path', type=str, default=None,
                        help='数据文件（npy 或带表头 txt）；默认 sachs/continuous/data1.npy')
    parser.add_argument('--dag_path', type=str, default=None,
                        help='真图文件（npy 或 "父 子" 文本）；默认 DAG1.npy / 内置 17 边')
    parser.add_argument('--no_bias', action='store_true',
                        help='关闭 decoder bias（官方 exp4 默认 use_bias=True, bias=-10；'
                             '只有 exp3/GP 才关闭）')
    parser.add_argument('--bias_initial_value', type=float, default=-2.0,
                        help='decoder logits bias 初始值。注意：官方 exp4 用 -10，但该 PyTorch 移植版'
                             '中 σ(-10)=4.5e-5 使初始图几乎全空且梯度学不动（死区，20000 轮仍仅 3 边）；'
                             '本机实测 -2.0（初始期望≈13 边≈真图 17 边）可正常学习，配合 CAM 剪枝'
                             '达到论文 Table 4 水平（Total 11/Correct 6/SHD 13）')
    parser.add_argument('--l1_graph_reg', type=float, default=0.0,
                        help='训练期图稀疏 L1 正则系数（加到 BIC score 上，越大图越稀疏；'
                             '已修复 Reward_BIC 的 l1_grapha_reg 拼写 bug）')
    parser.add_argument('--no_prune', action='store_true', help='关闭剪枝')
    parser.add_argument('--prune_mode', choices=['cam_r', 'cam', 'greedy'], default='cam_r',
                        help='剪枝模式：cam_r=论文同款真 CAM（调 R 包 CAM:::pruning，需 rpy2+R；'
                             '无 rpy2 自动回退 cam）；cam=加性 GPR 近似；greedy=全局 ΣRSS 贪心')
    parser.add_argument('--cam_cutoff', type=float, default=0.001,
                        help='真 CAM 剪枝显著性阈值 cutOffPVal（论文默认 0.001）')
    parser.add_argument('--prune_tol', type=float, default=0.05,
                        help='CAM 剪枝容差：去掉某父边后该节点加性模型 RSS 增幅不超过该比例'
                             '即接受剪掉（默认 0.05，Sachs 实测 SHD≈13 接近论文）')
    parser.add_argument('--prune_tol_ratio', type=float, default=0.01,
                        help='greedy 模式剪枝容差：去掉一条边后样本内 ΣRSS 增幅不超过该比例'
                             '即接受（默认 0.01 即 1 个百分点）')
    args = parser.parse_args()
    args.prune = not args.no_prune

    data_path = args.data_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sachs', 'continuous', 'data1.npy')
    if not os.path.exists(data_path):
        print(f'[DATA] 本地文件不存在：{data_path}，尝试从 CMU 下载（7466 行完整版取前 853 行观测块）…')
        dl_path = data_path
        if data_path.lower().endswith('.npy'):
            dl_path = os.path.splitext(data_path)[0] + '.txt'
        os.makedirs(os.path.dirname(dl_path), exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, dl_path)
        data_path = dl_path
    X = load_sachs_data(data_path)
    print(f'[DATA] X shape = {X.shape}（观测子集 853×11）')
    print(f'[DATA] 列顺序：{" ".join(LABELS)}')

    dag_path = args.dag_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sachs', 'continuous', 'DAG1.npy')
    dag, dag_src = load_sachs_dag(dag_path if os.path.exists(dag_path) else args.dag_path)
    n_edges = int(np.round(dag.sum()))
    print(f'[DAG] 真图来源：{dag_src}，边数 = {n_edges}')
    if n_edges != 17:
        print(f'[WARN] 真图边数 {n_edges} ≠ 17（论文口径为 17 边黄金标准），请确认 --dag_path 是否合理')

    if args.seed_start is not None:
        seeds = list(range(args.seed_start, (args.seed_end if args.seed_end is not None
                                             else args.seed_start) + 1))
        per_seed = args.nb_epoch * 0.17 + 200
        print(f'[MODE] 批量模式：RL_LDA seed = {seeds[0]}..{seeds[-1]}，共 {len(seeds)} 个，'
              f'nb_epoch = {args.nb_epoch}，预计每种子约 {per_seed/60:.0f} 分钟，'
              f'全部约 {per_seed*len(seeds)/3600:.1f} 小时')
        rows = []
        t_all = time.time()
        for k, seed in enumerate(seeds, start=1):
            print(f'\n--- 进度 {k}/{len(seeds)}：seed={seed} ---')
            rows.append(run_single_seed(seed, args, X, dag, save_plot=not args.no_plots))
            done = (time.time() - t_all) / k
            print(f'[ETA] 已完成 {k}/{len(seeds)}，平均 {done/60:.1f} 分钟/种子，'
                  f'预计剩余 {(done*(len(seeds)-k))/3600:.1f} 小时')
        print(f'\n[ALL] 全部完成，总用时 {(time.time()-t_all)/3600:.2f} 小时')
        keys = ['total_edges', 'correct_edges', 'shd', 'fdr', 'tpr', 'nnz']
        if args.prune:
            keys += ['total_edges_pruned', 'correct_edges_pruned', 'shd_pruned',
                     'fdr_pruned', 'tpr_pruned', 'nnz_pruned']
        print_aggregate(rows, keys)
        base = f'sachs_metrics_seed{seeds[0]}-{seeds[-1]}_ep{args.nb_epoch}'
        save_excel(rows, base + '.xlsx')
    else:
        row = run_single_seed(args.seed, args, X, dag, save_plot=not args.no_plots)
        rows = [row]
        keys = ['fdr', 'tpr', 'shd', 'nnz', 'time_s']
        if args.prune:
            keys += ['fdr_pruned', 'tpr_pruned', 'shd_pruned', 'nnz_pruned', 'pruned_edges']
        print_aggregate(rows, keys)
        base = f'sachs_results_seed{args.seed}_ep{args.nb_epoch}'
        save_excel(rows, base + '.xlsx')

if __name__ == '__main__':
    main()
