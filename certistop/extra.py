"""Dublin leave-one-sport-out experiment on the reported splits (seeds 0-19). Predictor, score selection and test
order come from three sports. The threshold is certified either on calibration tests of the other three sports
(transfer) or on calibration tests of the held-out sport (recalibrated), and tests of the held-out sport are scored.
Reported for the budget guarantee (interval width) and for per-stop CertiStop.
Usage: python certistop/extra.py --jobs 10"""
import argparse, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import load, split_ids, add_traj, GATE, ETA, N_HALVES
from models import Predictor
from core import first_stop, ltt_select, ltt_select_ordered, training_order
from sklearn.model_selection import GroupKFold

SPORTS = {0: 'Cycling', 1: 'Running', 2: 'Rowing', 3: 'Kayak'}
CANDS = {'width': ('width', 1), 'hr_frac': ('hr_frac', -1), 'time': ('tsec', -1), 'lactate': ('bla', -1),
         'stability': ('dy_abs_mean3', 1), 'since_gate': ('k_gate', -1)}


def prep(D, yhat, width):
    X = add_traj(D, yhat); X['err'] = X.yhat - X.y; X['width'] = width
    el = (X.hr_frac >= GATE).values
    X['k_gate'] = pd.Series(el.astype(int), index=X.index).groupby(X.ID_test.values, sort=False).cumsum().values - 1
    X['_el'] = el
    return X


def score(X, c):
    col, sg = CANDS[c]
    return np.where(X._el.values, np.nan_to_num(sg * X[col].values, nan=np.inf), np.inf)


def rec(out, X, s, lam, dlt, **kw):
    o = first_stop(X, s, [lam]); st = o["stopped"][:, 0]; bad = (np.abs(o["err"][:, 0]) > dlt) & st
    kw.update(n_test=X.ID_test.nunique(), risk=bad.mean(), n_stop=int(st.sum()), n_bad=int(bad.sum()), early=st.mean(), saved_s=o['saved'][:, 0].mean())
    out.append(kw)


def one(rep):
    P, FEATS = load('dublin'); tr, ca, te = split_ids(P, rep); out = []
    TR0, CAL0, TE0 = (P[P.ID.isin(s)].copy() for s in (tr, ca, te))
    feats = FEATS['full']
    for sp in range(4):
        TRo, CALo = TR0[TR0.sport != sp], CAL0[CAL0.sport != sp]
        CALs, TEs = CAL0[CAL0.sport == sp], TE0[TE0.sport == sp]
        if TEs.ID_test.nunique() < 5:
            continue
        m = Predictor(feats, True, seed=rep).fit(TRo)
        ql = Predictor(feats, True, seed=rep, quantile=0.1).fit(TRo); qh = Predictor(feats, True, seed=rep, quantile=0.9).fit(TRo)
        oof, wlo, whi = (np.zeros(len(TRo)) for _ in range(3))
        for a, b in GroupKFold(5).split(TRo, groups=TRo.ID):
            A, B = TRo.iloc[a], TRo.iloc[b]
            oof[b] = Predictor(feats, True, seed=rep).fit(A).predict(B)
            wlo[b] = Predictor(feats, True, seed=rep, quantile=0.1).fit(A).predict(B)
            whi[b] = Predictor(feats, True, seed=rep, quantile=0.9).fit(A).predict(B)
        TRx = prep(TRo, oof, whi - wlo)
        CALx, CALsx, TEx = (prep(D, m.predict(D), qh.predict(D) - ql.predict(D)) for D in (CALo, CALs, TEs))
        grids = {c: np.unique(np.quantile(score(TRx, c)[np.isfinite(score(TRx, c))], np.linspace(0, 1, 401))) for c in CANDS}
        ids = np.array(sorted(TRx.ID.unique()), dtype=object); hr = np.random.default_rng(8000 + rep)
        halves = [TRx.ID.isin(set(hr.permutation(ids)[:len(ids) // 2])).values for _ in range(N_HALVES)]
        for delta in (5.0,):
            # budget guarantee, interval width
            g = grids['width']; s_tr = score(TRx, 'width')
            for mode, C in (('transfer', CALx), ('recalibrated', CALsx)):
                j = ltt_select(C, score(C, 'width'), g, delta, 0.1, ETA, selective=False); lam = g[j] if j >= 0 else -np.inf
                rec(out, TEx, score(TEx, 'width'), lam, delta, rep=rep, sport=SPORTS[sp], delta=delta, guarantee='budget', alpha=0.1, mode=mode, choice='width', n_cal=C.ID_test.nunique())
            # per-stop CertiStop with score selection on the training tests of the other sports
            for alpha in (0.15, 0.20):
                best, bsave = None, -1.0
                for c in CANDS:
                    s = score(TRx, c); sv = []
                    for mA in halves:
                        order = training_order(TRx[mA], s[mA], grids[c], delta, alpha, selective=True)
                        j = ltt_select_ordered(TRx[~mA], s[~mA], grids[c], delta, alpha, ETA, order, selective=True)
                        sv.append(first_stop(TRx[~mA], s[~mA], [grids[c][j]])['saved'][:, 0].mean() if j >= 0 else 0.0)
                    if np.mean(sv) > bsave:
                        best, bsave = c, float(np.mean(sv))
                g = grids[best]; order = training_order(TRx, score(TRx, best), g, delta, alpha, selective=True)
                for mode, C in (('transfer', CALx), ('recalibrated', CALsx)):
                    j = ltt_select_ordered(C, score(C, best), g, delta, alpha, ETA, order, selective=True); lam = g[j] if j >= 0 else -np.inf
                    rec(out, TEx, score(TEx, best), lam, delta, rep=rep, sport=SPORTS[sp], delta=delta, guarantee='per_stop', alpha=alpha, mode=mode, choice=best, n_cal=C.ID_test.nunique())
    return pd.DataFrame(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', type=int, default=8); a = ap.parse_args()
    from joblib import Parallel, delayed
    R = pd.concat(Parallel(n_jobs=a.jobs, verbose=5)(delayed(one)(r) for r in range(20)))
    os.makedirs('results/extra', exist_ok=True); R.to_csv('results/extra/dublin_loso.csv', index=False)
    pd.set_option('display.width', 220)
    G = R.groupby(['guarantee', 'alpha', 'sport', 'mode']).agg(n_cal=('n_cal', 'mean'), n_test=('n_test', 'mean'), risk=('risk', 'mean'),
                                                                 n_stop=('n_stop', 'sum'), n_bad=('n_bad', 'sum'), early=('early', 'mean'), saved_s=('saved_s', 'mean'))
    G['sel_pooled'] = G.n_bad / G.n_stop.clip(lower=1)
    print(G.round(3).to_string())
