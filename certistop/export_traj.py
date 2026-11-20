"""Export per-decision test rows of one repetition (estimate, 80 % interval, certified width threshold) for plotting
example trajectories. Uses the saved models of that repetition. Usage: python certistop/export_traj.py --ds malaga --rep 0"""
import argparse, os, sys
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import load, split_ids

ap = argparse.ArgumentParser(); ap.add_argument('--ds', required=True); ap.add_argument('--rep', type=int, default=0)
a = ap.parse_args()
P, FEATS = load(a.ds)
tr, ca, te = split_ids(P, a.rep)
TE = P[P.ID.isin(te)].copy()
S = pd.read_csv(f'results/{a.ds}/stop_summary.csv')
out = []
for inp in ('full', 'hr'):
    main = joblib.load(f'weights/{a.ds}/rep{a.rep:02d}/{inp}_predictor.joblib')
    qlo, qhi = joblib.load(f'weights/{a.ds}/rep{a.rep:02d}/{inp}_quantiles.joblib')
    D = TE[['ID_test', 'tsec', 'Tsec', 'y', 'hr', 'hr_frac', 'hrmax']].copy()
    D['input'] = inp; D['yhat'] = main.predict(TE); D['qlo'] = qlo.predict(TE); D['qhi'] = qhi.predict(TE)
    for d in (3.5, 5.0):
        for r in ('width', 'hr_frac', 'time'):
            lam = S[(S.rep == a.rep) & (S.input == inp) & (S.delta == d) & (S.alpha == 0.1) & (S.rule == r)].lam.values[0]
            D[f'lam_{r}_d{d}'] = lam
    out.append(D)
pd.concat(out).to_parquet(f'results/{a.ds}/traj_rep{a.rep:02d}.parquet')
print('ok', len(TE))
