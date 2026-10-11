import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rl_lda'))

import argparse
import os

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from rl_lda.evaluation import MetricsDAG

CAM = importr('CAM')
importr('Matrix')

def cam_prune(X, est, cutoff=0.001, tmpdir='cam_tmp'):
    
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--est_path', default=r'sachs_est_seed1_ep20000.npy')
    ap.add_argument('--data_path', default=r'sachs/continuous/data1.npy')
    ap.add_argument('--dag_path', default=r'sachs/continuous/DAG1.npy')
    ap.add_argument('--cutoff', default='0.001,0.01,0.05,0.1',
                    help='逗号分隔的 cutOffPVal 阈值列表')
    ap.add_argument('--tmpdir', default='cam_tmp')
    args = ap.parse_args()

    from sklearn.preprocessing import StandardScaler
    X = np.load(args.data_path).astype(np.float64)
    Xs = StandardScaler().fit_transform(X)
    est = np.load(args.est_path)
    dag = np.load(args.dag_path)

    print(f'[DATA] X {X.shape}, est {est.shape} ({int(est.sum())} 边), true DAG {int(dag.sum())} 边')

    for cutoff in [float(c) for c in args.cutoff.split(',')]:
        Gp = cam_prune(Xs, est, cutoff=cutoff, tmpdir=args.tmpdir)
        m = MetricsDAG(Gp, dag).metrics
        correct = int((Gp * dag).sum())
        print(f'[CAM cutOffPVal={cutoff:g}] Total Edges={int(Gp.sum())}, '
              f'Correct Edges={correct}, SHD={int(m["shd"])} '
              f'(FDR={m["fdr"]:.4f}, TPR={m["tpr"]:.4f})')
        np.save(f'cam_pruned_cutoff{cutoff}.npy', Gp)

if __name__ == '__main__':
    main()
