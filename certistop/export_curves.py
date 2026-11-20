"""Risk-coverage curves of each stop score on the test splits: for thresholds spanning the observed score range,
the share of tests stopped early, the selective risk among them and the mean time saved. Uses the saved models.
Usage: python certistop/export_curves.py --ds malaga"""
import argparse, os, sys
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import load, split_ids, add_traj, GATE, first_eligible_random
from core import first_stop

ap = argparse.ArgumentParser(); ap.add_argument('--ds', required=True); ap.add_argument('--reps', type=int, default=20)
a = ap.parse_args()
P, FEATS = load(a.ds)
rows = []
for rep in range(a.reps):
    tr, ca, te = split_ids(P, rep)
    TE = P[P.ID.isin(te)].copy()
    for inp in ('full', 'hr'):
        main = joblib.load(f'weights/{a.ds}/rep{rep:02d}/{inp}_predictor.joblib')
        ql, qh = joblib.load(f'weights/{a.ds}/rep{rep:02d}/{inp}_quantiles.joblib')
        X = add_traj(TE, main.predict(TE)); X['err'] = X.yhat - X.y; X['width'] = qh.predict(TE) - ql.predict(TE)
        el = (X.hr_frac >= GATE).values
        kg = pd.Series(el.astype(int), index=X.index).groupby(X.ID_test.values, sort=False).cumsum().values - 1
        sc = {'width': X.width.values, 'hr_frac': -X.hr_frac.values, 'time': -X.tsec.values, 'stability': X.dy_abs_mean3.values, 'since_gate': -kg,
              'random': first_eligible_random(X, el, np.random.default_rng(9000 + rep))}
        if inp == 'full':
            if a.ds == 'malaga':
                sc.update(rer=-X.rer.values, plateau=X.vo2_recent_slope.values)
            else:
                sc.update(lactate=-X.bla.values)
        for nm, s in sc.items():
            s = np.where(el, np.nan_to_num(s, nan=np.inf), np.inf)
            fin = s[np.isfinite(s)]
            lams = np.unique(np.quantile(fin, np.linspace(0, 1, 101)))
            o = first_stop(X, s, lams)
            for d in (3.5, 5.0):
                st = o['stopped']; bad = (np.abs(o['err']) > d) & st
                for j in range(len(lams)):
                    n = st[:, j].sum()
                    rows.append(dict(rep=rep, input=inp, delta=d, score=nm, j=j, coverage=st[:, j].mean(),
                                     sel_risk=bad[:, j].sum() / n if n else np.nan, saved_s=o['saved'][:, j].mean()))
    print('rep', rep, flush=True)
pd.DataFrame(rows).to_parquet(f'results/{a.ds}/riskcov.parquet')
