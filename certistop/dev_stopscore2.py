"""Second design study on development splits (seeds 90-99): temporal smoothing of the stop score, a
cross-fitted residual-quantile score and a combination of the learned risk with the interval width."""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGR, HistGradientBoostingClassifier as HGC
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import load, split_ids, Predictor, add_traj, TRAJ
from core import first_stop, ltt_select


def smooth(D, s, k=2):
    return pd.Series(s, index=D.index).groupby(D.ID_test.values, sort=False).transform(lambda v: v.rolling(k, min_periods=1).max()).values


def ecdf(train_scores):
    srt = np.sort(train_scores)
    return lambda s: np.searchsorted(srt, s, side='right') / len(srt)


def one(ds, rep):
    P, FEATS = load(ds); tr, ca, te = split_ids(P, rep); out = []
    for inp, feats in FEATS.items():
        dt = inp == 'full'
        TR, CAL, TE = (P[P.ID.isin(s)].copy() for s in (tr, ca, te))
        oof, wlo, whi = (np.zeros(len(TR)) for _ in range(3))
        for a, b in GroupKFold(5).split(TR, groups=TR.ID):
            A, B = TR.iloc[a], TR.iloc[b]
            oof[b] = Predictor(feats, dt, seed=rep).fit(A).predict(B)
            wlo[b] = Predictor(feats, dt, seed=rep, quantile=0.1).fit(A).predict(B)
            whi[b] = Predictor(feats, dt, seed=rep, quantile=0.9).fit(A).predict(B)
        main = Predictor(feats, dt, seed=rep).fit(TR)
        ql = Predictor(feats, dt, seed=rep, quantile=0.1).fit(TR); qh = Predictor(feats, dt, seed=rep, quantile=0.9).fit(TR)
        TRx = add_traj(TR, oof); TRx['width'] = whi - wlo
        CALx = add_traj(CAL, main.predict(CAL)); CALx['width'] = qh.predict(CAL) - ql.predict(CAL)
        TEx = add_traj(TE, main.predict(TE)); TEx['width'] = qh.predict(TE) - ql.predict(TE)
        for D in (TRx, CALx, TEx): D['err'] = D.yhat - D.y
        cols = feats + [c for c in TRAJ if c in TRx] + ['width']
        for delta in (3.5, 5.0):
            clf = HGC(max_iter=300, learning_rate=0.05, min_samples_leaf=30, l2_regularization=1.0, random_state=rep).fit(TRx[cols], (TRx.err.abs() > delta).astype(int))
            rq = HGR(loss='quantile', quantile=0.8, max_iter=300, learning_rate=0.05, min_samples_leaf=30, l2_regularization=1.0, random_state=rep).fit(TRx[cols], TRx.err.abs())
            sc = {}
            for nm, D in (('tr', TRx), ('cal', CALx), ('te', TEx)):
                pl = clf.predict_proba(D[cols])[:, 1]; w = D.width.values; r = rq.predict(D[cols])
                sc.setdefault('learned', {})[nm] = pl
                sc.setdefault('width', {})[nm] = w
                sc.setdefault('resq80', {})[nm] = r
                sc.setdefault('learned_sm2', {})[nm] = smooth(D, pl, 2)
                sc.setdefault('width_sm2', {})[nm] = smooth(D, w, 2)
                sc.setdefault('resq80_sm2', {})[nm] = smooth(D, r, 2)
            Fl, Fw = ecdf(sc['learned']['tr']), ecdf(sc['width']['tr'])
            sc['combo'] = {k: 0.5 * (Fl(sc['learned'][k]) + Fw(sc['width'][k])) for k in ('tr', 'cal', 'te')}
            sc['combo_sm2'] = {k: smooth(D, sc['combo'][k], 2) for k, D in (('tr', TRx), ('cal', CALx), ('te', TEx))}
            for nm, s in sc.items():
                grid = np.unique(np.quantile(s['tr'], np.linspace(0, 1, 401)))
                for alpha in (0.05, 0.1, 0.2):
                    j = ltt_select(CALx, s['cal'], grid, delta, alpha, 0.1, selective=False); lam = grid[j] if j >= 0 else -np.inf
                    o = first_stop(TEx, s['te'], [lam]); st = o['stopped'][:, 0]
                    out.append(dict(ds=ds, rep=rep, input=inp, delta=delta, alpha=alpha, score=nm, risk=((np.abs(o['err'][:, 0]) > delta) & st).mean(),
                                    saved_s=o['saved'][:, 0].mean()))
    return pd.DataFrame(out)


if __name__ == '__main__':
    from joblib import Parallel, delayed
    R = pd.concat(Parallel(n_jobs=20, verbose=2)(delayed(one)(ds, r) for ds in ('malaga', 'dublin') for r in range(90, 100)))
    R.to_csv('results/dev/stopscore_designs2.csv', index=False)
    pd.set_option('display.width', 250)
    print(R.groupby(['ds', 'input', 'score', 'delta', 'alpha']).saved_s.mean().unstack([3, 4]).round(1).to_string())
    print(R.groupby(['ds', 'input', 'score']).risk.mean().unstack([0, 1]).round(3).to_string())
