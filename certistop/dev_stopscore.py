"""Design study for the stop score on development splits (seeds 90-99, never used for reported results).
Compares classifier/regressor designs for the risk score by calibration-set AUC and by certified time saved."""
import os, sys, argparse
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGR, HistGradientBoostingClassifier as HGC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import load, split_ids, Predictor, add_traj, TRAJ
from core import first_stop, ltt_select


def one(ds, rep):
    P, FEATS = load(ds)
    tr, ca, te = split_ids(P, rep)
    out = []
    for inp, feats in FEATS.items():
        dt = inp == 'full'
        TR, CAL, TE = (P[P.ID.isin(s)].copy() for s in (tr, ca, te))
        oof = np.zeros(len(TR)); wlo = np.zeros(len(TR)); whi = np.zeros(len(TR))
        for a, b in GroupKFold(5).split(TR, groups=TR.ID):
            oof[b] = Predictor(feats, dt, seed=rep).fit(TR.iloc[a]).predict(TR.iloc[b])
            wlo[b] = Predictor(feats, dt, seed=rep, quantile=0.1).fit(TR.iloc[a]).predict(TR.iloc[b])
            whi[b] = Predictor(feats, dt, seed=rep, quantile=0.9).fit(TR.iloc[a]).predict(TR.iloc[b])
        main = Predictor(feats, dt, seed=rep).fit(TR)
        ql = Predictor(feats, dt, seed=rep, quantile=0.1).fit(TR); qh = Predictor(feats, dt, seed=rep, quantile=0.9).fit(TR)
        TRx = add_traj(TR, oof); TRx['width'] = whi - wlo
        CALx = add_traj(CAL, main.predict(CAL)); CALx['width'] = qh.predict(CAL) - ql.predict(CAL)
        TEx = add_traj(TE, main.predict(TE)); TEx['width'] = qh.predict(TE) - ql.predict(TE)
        for D in (TRx, CALx, TEx): D['err'] = D.yhat - D.y
        base = feats + [c for c in TRAJ if c in TRx]
        fsets = {'f': feats, 'f+traj': base, 'f+traj+w': base + ['width']}
        for delta in (3.5, 5.0):
            yl = (TRx.err.abs() > delta).astype(int)
            designs = {}
            for fs_name, cols in fsets.items():
                designs[f'hgc_default|{fs_name}'] = ('c', HGC(max_iter=300, learning_rate=0.05, min_samples_leaf=30, l2_regularization=1.0, random_state=rep), cols)
                designs[f'hgc_reg|{fs_name}'] = ('c', HGC(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=80, l2_regularization=5.0, random_state=rep), cols)
                designs[f'hgr_abs|{fs_name}'] = ('r', HGR(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=80, l2_regularization=5.0, random_state=rep), cols)
                designs[f'logit|{fs_name}'] = ('c', make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)), cols)
            for nm, (kind, m, cols) in designs.items():
                if kind == 'c':
                    m.fit(TRx[cols], yl); f = lambda D: m.predict_proba(D[cols])[:, 1]
                else:
                    m.fit(TRx[cols], TRx.err.abs()); f = lambda D: m.predict(D[cols])
                s_tr, s_ca, s_te = f(TRx), f(CALx), f(TEx)
                auc = roc_auc_score((CALx.err.abs() > delta).astype(int), s_ca)
                grid = np.unique(np.quantile(s_tr, np.linspace(0, 1, 401)))
                for alpha in (0.1, 0.2):
                    j = ltt_select(CALx, s_ca, grid, delta, alpha, 0.1, selective=False); lam = grid[j] if j >= 0 else -np.inf
                    o = first_stop(TEx, s_te, [lam]); st = o['stopped'][:, 0]
                    out.append(dict(ds=ds, rep=rep, input=inp, delta=delta, alpha=alpha, design=nm, auc_cal=auc,
                                    risk=((np.abs(o['err'][:, 0]) > delta) & st).mean(), saved_s=o['saved'][:, 0].mean(), early=st.mean()))
    return pd.DataFrame(out)


if __name__ == '__main__':
    from joblib import Parallel, delayed
    jobs = [(ds, r) for ds in ('malaga', 'dublin') for r in range(90, 100)]
    R = pd.concat(Parallel(n_jobs=20, verbose=5)(delayed(one)(ds, r) for ds, r in jobs))
    os.makedirs('results/dev', exist_ok=True); R.to_csv('results/dev/stopscore_designs.csv', index=False)
    pd.set_option('display.width', 250)
    print(R.groupby(['ds', 'input', 'delta', 'alpha', 'design'])[['auc_cal', 'risk', 'saved_s']].mean().unstack([0, 1, 2, 3]).round(3).to_string())
