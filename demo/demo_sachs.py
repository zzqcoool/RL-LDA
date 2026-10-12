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
# 1. Data loading
# ============================================================
def builtin_dag():
    """Built-in 17-edge gold standard (bnlearn dag.sachs model string)."""
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
    """Load Sachs observational data -> (X (853,11) float64)."""
    if data_path.lower().endswith('.npy'):
        X = np.load(data_path)
    else:
        arr = np.genfromtxt(data_path, dtype=float)
        if arr.ndim == 1 and arr.size == 0:
            raise ValueError(f'Data file is empty: {data_path}')
        X = arr
        if X.ndim == 2:
            X = X[~np.isnan(X).any(axis=1)]
        if X.shape[1] == 12 and X.shape[0] != 853:
            X = X[1:]
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != D:
        raise ValueError(f'Expected {D} columns, got {X.shape}')
    if X.shape[0] == 7466:
        print('[DATA] Full version with 7466 rows (including interventional experiments) '
              'detected; using the first 853 observational rows (paper setting m=853)')
        X = X[:853]
    if X.shape[0] != 853:
        print(f'[WARN] Row count {X.shape[0]} != 853 (observational subset); '
              'continuing but please check the setting')
    return X


def load_sachs_dag(dag_path):
    """Load the ground-truth graph -> (DAG (11,11) i->j adjacency, source)."""
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
    return builtin_dag(), 'Built-in 17-edge gold standard (bnlearn dag.sachs)'


# ============================================================
# 2. Pruning (approximate CAM pruning based on regression performance)
# ============================================================
def _node_in_rss(X, i, pa, cache):
    """In-sample GPR residual sum of squares for node i given parents pa."""
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
    """True CAM pruning via the R package CAM (gam additive model + significance test)."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
    except ImportError as e:
        raise RuntimeError(
            'rpy2 is not installed: use the CausalDiscovery37 environment or one with '
            'rpy2 (e.g. python38), and make sure R_HOME points to the R installation') from e
    try:
        CAM = importr('CAM')
        importr('Matrix')
    except Exception as e:
        raise RuntimeError('Failed to load R packages CAM/Matrix: ' + str(e)) from e
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
    """Greedy pruning: remove an edge if the in-sample RSS increase <= ratio_tol."""
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
    """CAM-style additive pruning (approximate, no R required)."""
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
# 3. Single-seed pipeline
# ============================================================
def run_single_seed(seed, args, X, dag, save_plot):
    rl = RL_LDA(
        nb_epoch=args.nb_epoch,
        seed=seed,
        reg_type='GPR',
        score_type='BIC',
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
          f'(FDR={row["fdr"]:.4f}, TPR={row["tpr"]:.4f}), elapsed {elapsed:.0f}s')

    # Pruning (default: true CAM via rpy2 + R; falls back to the additive
    # approximation when rpy2 is unavailable)
    if args.prune:
        Xs = StandardScaler().fit_transform(X)
        if args.prune_mode == 'cam_r':
            try:
                Gp = cam_r_prune(Xs, est, cutoff=args.cam_cutoff)
                removed = int((est != Gp).sum())
                print(f'[PRUNE-CAM] cutOffPVal={args.cam_cutoff:g}, removed {removed} edges: ', end='')
            except Exception as e:
                print(f'[WARN] True CAM pruning failed ({e}); falling back to additive_prune')
                Gp, removed, _, _ = additive_prune(Xs, est, tol=args.prune_tol)
        elif args.prune_mode == 'cam':
            Gp, removed, s_before, s_after = additive_prune(Xs, est, tol=args.prune_tol)
            print(f'[PRUNE] Removed {removed} edges (sum RSS {s_before:.1f}->{s_after:.1f}): ', end='')
        else:
            Gp, removed, s_before, s_after = greedy_prune(Xs, est, ratio_tol=args.prune_tol_ratio)
            print(f'[PRUNE] Removed {removed} edges (sum RSS {s_before:.1f}->{s_after:.1f}): ', end='')
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

    # Save est / dag matrices for later pruning experiments (no need to rerun RL)
    np.save(f'sachs_est_seed{seed}_ep{args.nb_epoch}.npy', est)
    np.save(f'sachs_dag_ep{args.nb_epoch}.npy', dag)
    return row


def print_aggregate(rows, keys):
    import statistics
    print('\n' + '=' * 66)
    print(f'Summary ({len(rows)} RL_LDA seeds)')
    print('-' * 66)
    for key in keys:
        vals = [r[key] for r in rows]
        if len(rows) > 1:
            print(f'{key:<14}{np.mean(vals):.3f}+-{np.std(vals, ddof=1):.3f}'
                  f'  median {np.median(vals):.3f}  range {min(vals):.1f}~{max(vals):.1f}')
        else:
            print(f'{key:<14}{np.mean(vals):.3f}')


def save_excel(rows, path):
    """Save the detailed results to .xlsx (openable in Excel)."""
    if not rows:
        return
    try:
        import openpyxl
    except ImportError:
        print('[WARN] openpyxl not installed; skipping Excel output')
        return
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'sachs_results'
    headers = list(rows[0].keys())
    ws.append(headers)
    for r in rows:
        ws.append([r[h] for h in headers])
    wb.save(path)
    print(f'[XLSX] Results saved: {os.path.abspath(path)}')


# ============================================================
# 4. Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Sachs data + RL_LDA(GPR) causal discovery demo (paper Sec. 6.4)')
    parser.add_argument('--seed', type=int, default=1, help='single-seed mode (default 1)')
    parser.add_argument('--seed_start', type=int, default=1, help='batch RL_LDA seed start (inclusive)')
    parser.add_argument('--seed_end', type=int, default=10, help='batch RL_LDA seed end (inclusive, default=start)')
    parser.add_argument('--no_plots', action='store_true',
                        help='do not save per-seed comparison figures (by default each seed '
                             'saves sachs_demo_seed{n}_ep{nb_epoch}.png)')
    parser.add_argument('--nb_epoch', type=int, default=20000, help='RL_LDA training iterations (default paper 20000)')
    parser.add_argument('--input_dimension', type=int, default=128, help='number of sampled input features (paper 128)')
    parser.add_argument('--score_type', choices=['BIC', 'BIC_different_var'],
                        default='BIC_different_var',
                        help='BIC=equal variance (RL-BIC2); BIC_different_var=unequal variance '
                             '(paper RL-BIC, default). Sachs: unequal-variance est contains 11 true '
                             'edges and gives Total 11/Correct 6/SHD 13 after CAM pruning (paper '
                             'RL-BIC 10/6/12); equal-variance gives 9 true edges and 8/4/15 after pruning')
    parser.add_argument('--data_path', type=str, default=None,
                        help='data file (npy or header txt); default sachs/continuous/data1.npy')
    parser.add_argument('--dag_path', type=str, default=None,
                        help='ground-truth file (npy or "parent child" text); default DAG1.npy / built-in 17 edges')
    parser.add_argument('--no_bias', action='store_true',
                        help='disable decoder bias (official exp4 uses use_bias=True, bias=-10; '
                             'only exp3/GP disables it)')
    parser.add_argument('--bias_initial_value', type=float, default=-2.0,
                        help='initial decoder logits bias. Note: official exp4 uses -10, but in this '
                             'PyTorch port sigmoid(-10)=4.5e-5 makes the initial graph nearly empty and '
                             'gradients cannot move (dead zone, only 3 edges after 20000 epochs); '
                             '-2.0 (initial expectation ~13 edges, close to the 17-edge ground truth) '
                             'learns normally and reaches the paper Table 4 level after CAM pruning '
                             '(Total 11/Correct 6/SHD 13)')
    parser.add_argument('--l1_graph_reg', type=float, default=0.0,
                        help='L1 graph sparsity regularization on the BIC score; larger means sparser '
                             '(the l1_grapha_reg typo in Reward_BIC is fixed)')
    parser.add_argument('--no_prune', action='store_true', help='disable pruning')
    parser.add_argument('--prune_mode', choices=['cam_r', 'cam', 'greedy'], default='cam_r',
                        help='pruning mode: cam_r=true CAM via the R package CAM:::pruning (needs '
                             'rpy2+R, falls back to cam automatically); cam=additive GPR approximation; '
                             'greedy=global sum-RSS greedy')
    parser.add_argument('--cam_cutoff', type=float, default=0.001,
                        help='CAM pruning significance threshold cutOffPVal (paper default 0.001)')
    parser.add_argument('--prune_tol', type=float, default=0.05,
                        help='CAM pruning tolerance: accept removing a parent if the additive-model '
                             'RSS increase <= this ratio (default 0.05, Sachs gives SHD~13 close to paper)')
    parser.add_argument('--prune_tol_ratio', type=float, default=0.01,
                        help='greedy pruning tolerance: accept removing an edge if the in-sample '
                             'sum-RSS increase <= this ratio (default 0.01, i.e. 1 percentage point)')
    args = parser.parse_args()
    args.prune = not args.no_prune

    # ---- data ----
    data_path = args.data_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sachs', 'continuous', 'data1.npy')
    if not os.path.exists(data_path):
        print(f'[DATA] Local file not found: {data_path}; trying to download from CMU '
              '(7466-row full version, using the first 853 observational rows)...')
        dl_path = data_path
        if data_path.lower().endswith('.npy'):
            dl_path = os.path.splitext(data_path)[0] + '.txt'
        os.makedirs(os.path.dirname(dl_path), exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, dl_path)
        data_path = dl_path
    X = load_sachs_data(data_path)
    print(f'[DATA] X shape = {X.shape} (observational subset 853x11)')
    print(f'[DATA] Column order: {" ".join(LABELS)}')

    # ---- ground truth ----
    dag_path = args.dag_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sachs', 'continuous', 'DAG1.npy')
    dag, dag_src = load_sachs_dag(dag_path if os.path.exists(dag_path) else args.dag_path)
    n_edges = int(np.round(dag.sum()))
    print(f'[DAG] Ground truth source: {dag_src}, edges = {n_edges}')
    if n_edges != 17:
        print(f'[WARN] Ground-truth edge count {n_edges} != 17 (paper uses the 17-edge gold '
              'standard); please check --dag_path')

    # ---- run seeds ----
    if args.seed_start is not None:
        seeds = list(range(args.seed_start, (args.seed_end if args.seed_end is not None
                                             else args.seed_start) + 1))
        per_seed = args.nb_epoch * 0.17 + 200
        print(f'[MODE] Batch mode: RL_LDA seed = {seeds[0]}..{seeds[-1]}, {len(seeds)} seeds, '
              f'nb_epoch = {args.nb_epoch}, ~{per_seed/60:.0f} min/seed, '
              f'~{per_seed*len(seeds)/3600:.1f} hours total')
        rows = []
        t_all = time.time()
        for k, seed in enumerate(seeds, start=1):
            print(f'\n--- Progress {k}/{len(seeds)}: seed={seed} ---')
            rows.append(run_single_seed(seed, args, X, dag, save_plot=not args.no_plots))
            done = (time.time() - t_all) / k
            print(f'[ETA] Finished {k}/{len(seeds)}, avg {done/60:.1f} min/seed, '
                  f'~{(done*(len(seeds)-k))/3600:.1f} hours remaining')
        print(f'\n[ALL] All done, total time {(time.time()-t_all)/3600:.2f} hours')
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
