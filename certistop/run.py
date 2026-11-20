"""Main experiment: anytime VO2max prediction, stop scores and risk-controlled calibration over repeated
participant-grouped splits (train 50 %, calibration 25 %, test 25 %).

Usage: python certistop/run.py --ds malaga --reps 20 --jobs 16
Writes results/<ds>/stop_summary.csv, stop_tests.parquet, pred_rows.parquet, cls_calib.csv and weights/<ds>/rep*/.
"""
import argparse, os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingRegressor as HGR, HistGradientBoostingClassifier as HGC
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import first_stop, ltt_select, ltt_select_ordered, training_order
from models import Predictor, hgr

META = ['ID', 'ID_test', 'tsec', 'Tsec', 'y', 'hrmax', 'step']
GAS_MALAGA = ['vo2', 'vo2_max_sofar', 'vco2', 've', 'rr', 'rer', 've_vo2', 've_vco2', 'excess_co2', 'o2pulse', 've_vo2_rise',
              've_vco2_rise', 't_since_vt1', 'vo2_at_vt1', 'vo2_hr_slope', 'hr_extrap', 'vo2_speed_slope', 'vo2_recent_slope']
LAB_DUBLIN = ['vo2', 'dvo2', 'vo2_rest', 'vo2_max_sofar', 'bla', 'dbla', 'bla_rest', 'vo2_hr_slope', 'hr_extrap',
              'vo2_load_slope', 'lbla_load_slope', 'o2pulse']
NONEX = ['age', 'sex', 'sexm', 'height', 'weight', 'bmi', 'sport']   # static covariates shared by all predictors
DELTAS = (3.5, 5.0)
ALPHAS = (0.05, 0.10, 0.20)
ALPHAS_SEL = (0.10, 0.15, 0.20)
AUTO_CANDS = ('width', 'time', 'hr_frac', 'rer', 'lactate', 'plateau', 'stability', 'since_gate')
SEL_RULES = ('width', 'learned', 'hr_frac', 'time', 'rer', 'lactate', 'plateau', 'stability', 'random', 'since_gate')
N_HALVES = 5
LIGHT = os.environ.get('CERTISTOP_LIGHT', '0') == '1'
ETA = 0.10
GATE = float(os.environ.get('CERTISTOP_GATE', '0.77'))   # stopping allowed once HR >= GATE x age-predicted HRmax (0.77: vigorous intensity)


def load(ds):
    P = pd.read_parquet(f'data/{ds}_prefix.parquet')
    if ds == 'malaga':
        P = P.rename(columns={'t': 'tsec', 'T': 'Tsec'})
        P['step'] = (P.tsec // 30).astype(int)
        lab = GAS_MALAGA
    else:
        lab = LAB_DUBLIN
    P = P[P.tsec < P.Tsec].sort_values(['ID_test', 'tsec']).reset_index(drop=True)
    feats_full = [c for c in P.columns if c not in META]
    feats_hr = [c for c in feats_full if c not in lab]
    return P, {'full': feats_full, 'hr': feats_hr}


def add_traj(D, yhat):
    """Causal trajectory features of the running estimate within each test."""
    D = D.copy(); D['yhat'] = yhat
    g = D.groupby('ID_test', sort=False).yhat
    D['dy1'] = D.yhat - g.shift(1); D['dy2'] = D.yhat - g.shift(2)
    D['dy_abs_mean3'] = D.dy1.abs().groupby(D.ID_test, sort=False).transform(lambda s: s.rolling(3, min_periods=1).mean())
    if 'vo2_max_sofar' in D:
        D['headroom'] = D.yhat - D.vo2_max_sofar
    return D


TRAJ = ['yhat', 'dy1', 'dy2', 'dy_abs_mean3', 'headroom']


def ece(p, y, bins=10):
    e, n = 0.0, len(p)
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return e


def split_ids(P, rep):
    ids = np.array(sorted(P.ID.unique()), dtype=object)
    rng = np.random.default_rng(1000 + rep); rng.shuffle(ids)
    n = len(ids); a, b = int(0.5 * n), int(0.75 * n)
    return set(ids[:a]), set(ids[a:b]), set(ids[b:])


def run_rep(ds, rep, P, FEATS, outdir, wdir):
    t0 = time.time()
    tr_ids, cal_ids, te_ids = split_ids(P, rep)
    TR0, CAL0, TE0 = (P[P.ID.isin(s)].copy() for s in (tr_ids, cal_ids, te_ids))
    summ, tests, preds, calib, timing = [], [], [], [], []
    os.makedirs(f'{wdir}/rep{rep:02d}', exist_ok=True)
    gcol = 'sex' if 'sex' in P else 'sexm'
    configs = [('full', FEATS['full'], True), ('hr', FEATS['hr'], False), ('full_nodelta', FEATS['full'], False)]
    if LIGHT:
        configs = configs[:1]
    for inp, feats, dt in configs:
        light = inp == 'full_nodelta'   # ablation of the residual target: only the main rules are evaluated (LIGHT restricts inputs and delta only)
        TR, CAL, TE = TR0.copy(), CAL0.copy(), TE0.copy()
        # ---- main anytime predictor, quantile models and cross-fitted training predictions
        t1 = time.time()
        main = Predictor(feats, dt, seed=rep).fit(TR)
        qlo = Predictor(feats, dt, seed=rep, quantile=0.1).fit(TR); qhi = Predictor(feats, dt, seed=rep, quantile=0.9).fit(TR)
        fit_s = time.time() - t1
        joblib.dump(main, f'{wdir}/rep{rep:02d}/{inp}_predictor.joblib', compress=3)
        joblib.dump((qlo, qhi), f'{wdir}/rep{rep:02d}/{inp}_quantiles.joblib', compress=3)
        oof, wlo, whi = (np.zeros(len(TR)) for _ in range(3))
        for a, b in GroupKFold(5).split(TR, groups=TR.ID):
            A, B = TR.iloc[a], TR.iloc[b]
            oof[b] = Predictor(feats, dt, seed=rep).fit(A).predict(B)
            wlo[b] = Predictor(feats, dt, seed=rep, quantile=0.1).fit(A).predict(B)
            whi[b] = Predictor(feats, dt, seed=rep, quantile=0.9).fit(A).predict(B)
        ins = main.predict(TR)
        TRx = add_traj(TR, oof); TRx['err'] = TRx.yhat - TRx.y; TRx['width'] = whi - wlo
        TRin = add_traj(TR, ins); TRin['err'] = TRin.yhat - TRin.y; TRin['width'] = qhi.predict(TR) - qlo.predict(TR)
        CALx = add_traj(CAL, main.predict(CAL)); CALx['err'] = CALx.yhat - CALx.y; CALx['width'] = qhi.predict(CAL) - qlo.predict(CAL)
        t1 = time.time()
        yte = main.predict(TE); wte = qhi.predict(TE) - qlo.predict(TE)
        inf_ms = (time.time() - t1) / len(TE) * 1000
        TEx = add_traj(TE, yte); TEx['err'] = TEx.yhat - TEx.y; TEx['width'] = wte
        timing.append(dict(rep=rep, input=inp, fit_s=fit_s, infer_ms_per_decision=inf_ms, n_train_rows=len(TR)))
        # ---- comparison predictors (accuracy only)
        pr = {'main': TEx.yhat.values}
        if not light:
            if dt:
                pr['gbm_direct'] = Predictor(feats, False, seed=rep).fit(TR).predict(TE)
            pr['ridge'] = Predictor(feats, False, kind='ridge').fit(TR).predict(TE)
            pr['nonexercise'] = Predictor([c for c in feats if c in NONEX], False, kind='ridge').fit(TR).predict(TE)
            if 'acsm_extrap' in TE:
                pr['acsm_extrap'] = TE.acsm_extrap.fillna(pd.Series(pr['nonexercise'], index=TE.index)).values
            if dt and 'hr_extrap' in TE:
                pr['hr_extrap'] = TE.hr_extrap.fillna(pd.Series(pr['nonexercise'], index=TE.index)).values
        for k, v in pr.items():
            preds.append(pd.DataFrame(dict(rep=rep, input=inp, model=k, ID_test=TE.ID_test.values, tsec=TE.tsec.values,
                                           Tsec=TE.Tsec.values, y=TE.y.values, hr_frac=TE.hr_frac.values, yhat=v)))
        # ---- interval end points, coverage of the 80 % interval, eligibility gate, decisions since the gate
        TEx['qlo'] = qlo.predict(TE); TEx['qhi'] = qhi.predict(TE)
        CALx['qlo'] = qlo.predict(CAL); CALx['qhi'] = qhi.predict(CAL)
        el = {k: (D.hr_frac >= GATE).values for k, D in (('tr', TRx), ('cal', CALx), ('te', TEx))}
        for k, D in (('tr', TRx), ('cal', CALx), ('te', TEx)):
            D['k_gate'] = pd.Series(el[k].astype(int), index=D.index).groupby(D.ID_test.values, sort=False).cumsum().values - 1
        inside = ((TEx.y >= TEx.qlo) & (TEx.y <= TEx.qhi)).values
        timing[-1].update(cov80_all=inside.mean(), cov80_elig=inside[el['te']].mean() if el['te'].any() else np.nan,
                          share_rows_elig=el['te'].mean())
        gate = lambda s, k: np.where(el[k], np.nan_to_num(s, nan=np.inf), np.inf)
        # ---- score definitions: column -> sign (stop when sign*value <= lambda)
        heur = {'width': ('width', 1), 'hr_frac': ('hr_frac', -1), 'since_gate': ('k_gate', -1)}
        if not light:
            heur.update({'time': ('tsec', -1), 'stability': ('dy_abs_mean3', 1)})
            if ds == 'malaga' and dt:
                heur.update({'rer': ('rer', -1), 'plateau': ('vo2_recent_slope', 1)})
            if ds == 'dublin' and dt:
                heur.update({'lactate': ('bla', -1)})
        traj = [c for c in TRAJ if c in TRx]
        sfeats = feats + traj + ['width']
        rng = np.random.default_rng(5000 + rep)
        cal_tests = np.array(CALx.ID_test.drop_duplicates().values, dtype=object)
        cal_sub = {n: set(rng.choice(cal_tests, size=min(n, len(cal_tests)), replace=False)) for n in (50, 100)}
        one_pp = set(CALx.drop_duplicates('ID_test').groupby('ID').ID_test.apply(lambda s: s.sample(1, random_state=rep).iloc[0]))
        rnd = {k: first_eligible_random(D, el[k], np.random.default_rng(7000 + rep + 100 * i)) for i, (k, D) in enumerate((('tr', TRx), ('cal', CALx), ('te', TEx)))}
        # participant-grouped halves of the training tests used to simulate certification for score selection
        tr_ids = np.array(sorted(TRx.ID.unique()), dtype=object); hrng = np.random.default_rng(8000 + rep)
        halves = []
        for _ in range(N_HALVES):
            pa = set(hrng.permutation(tr_ids)[:len(tr_ids) // 2]); halves.append(TRx.ID.isin(pa).values)
        for delta in ((5.0,) if LIGHT else DELTAS):
            lab = (TRx.err.abs() > delta).astype(int)
            variants = {'learned': (sfeats, TRx, lab)}                           # dedicated risk classifier
            if not light:
                variants['learned_basic'] = (['tsec', 'hr_frac'], TRx, lab)      # heuristic inputs only
            scores = {}
            for nm, (cols, Dfit, yl) in variants.items():
                clf = HGC(max_iter=300, learning_rate=0.05, min_samples_leaf=30, l2_regularization=1.0, random_state=rep).fit(Dfit[cols], yl)
                if nm == 'learned':
                    joblib.dump(clf, f'{wdir}/rep{rep:02d}/{inp}_riskclf_d{delta}.joblib', compress=3)
                    pt = clf.predict_proba(TEx[cols])[:, 1]; yt = (TEx.err.abs() > delta).values.astype(int); m_ = el['te']
                    calib.append(dict(rep=rep, input=inp, delta=delta, brier=np.mean((pt[m_] - yt[m_]) ** 2), ece=ece(pt[m_], yt[m_]),
                                      base_rate=yt[m_].mean(), auc=_auc(pt[m_], yt[m_]), auc_width=_auc(TEx.width.values[m_], yt[m_])))
                scores[nm] = {k: gate(clf.predict_proba(D[cols])[:, 1], k) for k, D in (('tr', TRx), ('cal', CALx), ('te', TEx))}
            for nm, (col, sg) in heur.items():
                scores[nm] = {k: gate(sg * D[col].values, k) for k, D in (('tr', TRx), ('cal', CALx), ('te', TEx))}
            if not light:
                scores['random'] = rnd                                           # no-information baseline (first eligible decision)
            grids = {}
            for nm, sc in scores.items():
                fin = sc['tr'][np.isfinite(sc['tr'])]
                grids[nm] = np.linspace(0, 1, 401) if nm == 'random' else np.unique(np.quantile(fin, np.linspace(0, 1, 401)))
            for nm, sc in scores.items():
                s_tr, s_cal, s_te = sc['tr'], sc['cal'], sc['te']
                grid = grids[nm]
                for alpha in ALPHAS:
                    j = ltt_select(CALx, s_cal, grid, delta, alpha, ETA, selective=False)
                    lam = grid[j] if j >= 0 else -np.inf
                    _record(summ, tests, rep, inp, delta, alpha, nm, lam, first_stop(TEx, s_te, [lam]), TEx, delta)
                    if light or nm != 'width':
                        continue
                    for n, ids in cal_sub.items():   # smaller calibration sets
                        msk = CALx.ID_test.isin(ids).values
                        j = ltt_select(CALx[msk], s_cal[msk], grid, delta, alpha, ETA, selective=False)
                        lam_n = grid[j] if j >= 0 else -np.inf
                        _record(summ, tests, rep, inp, delta, alpha, f'width_cal{n}', lam_n, first_stop(TEx, s_te, [lam_n]), TEx, delta)
                    msk = CALx.ID_test.isin(one_pp).values   # one calibration test per participant
                    j = ltt_select(CALx[msk], s_cal[msk], grid, delta, alpha, ETA, selective=False)
                    lam_n = grid[j] if j >= 0 else -np.inf
                    _record(summ, tests, rep, inp, delta, alpha, 'width_cal1pp', lam_n, first_stop(TEx, s_te, [lam_n]), TEx, delta)
                    if alpha == 0.10:         # sensitivity to the confidence level eta
                        for eta in (0.05, 0.20):
                            j = ltt_select(CALx, s_cal, grid, delta, alpha, eta, selective=False)
                            lam_e = grid[j] if j >= 0 else -np.inf
                            _record(summ, tests, rep, inp, delta, alpha, f'width_eta{int(eta * 100):02d}', lam_e, first_stop(TEx, s_te, [lam_e]), TEx, delta)
                    _record(summ, tests, rep, inp, delta, alpha, 'width_bysex', np.nan,
                            first_stop(TEx, bysex_scores(CALx, TEx, s_cal, s_te, grid, gcol, delta, alpha, selective=False), [0.0]), TEx, delta)
                # per-stop risk control, fixed-sequence LTT ordered by training p-values
                if nm in SEL_RULES:
                    for alpha in ALPHAS_SEL:
                        order = training_order(TRx, s_tr, grid, delta, alpha, selective=True)
                        j = ltt_select_ordered(CALx, s_cal, grid, delta, alpha, ETA, order, selective=True)
                        lam = grid[j] if j >= 0 else -np.inf
                        _record(summ, tests, rep, inp, delta, alpha, f'{nm}_sel', lam, first_stop(TEx, s_te, [lam]), TEx, delta)
            # ---- CertiStop: per-stop guarantee with the stop score selected on training tests. Each candidate is ordered on
            # one half of the training participants and certified on the other half; the mean saving over N_HALVES halvings
            # decides. Calibration labels are not used for the selection.
            cands = [c for c in AUTO_CANDS if c in scores]
            for alpha in ALPHAS_SEL:
                best, bsave = None, -1.0
                for c in cands:
                    s = scores[c]['tr']; sv = []
                    for mA in halves:
                        order = training_order(TRx[mA], s[mA], grids[c], delta, alpha, selective=True)
                        j = ltt_select_ordered(TRx[~mA], s[~mA], grids[c], delta, alpha, ETA, order, selective=True)
                        sv.append(first_stop(TRx[~mA], s[~mA], [grids[c][j]])['saved'][:, 0].mean() if j >= 0 else 0.0)
                    if np.mean(sv) > bsave:
                        best, bsave = c, float(np.mean(sv))
                g = grids[best]; sc = scores[best]
                order = training_order(TRx, sc['tr'], g, delta, alpha, selective=True)
                j = ltt_select_ordered(CALx, sc['cal'], g, delta, alpha, ETA, order, selective=True)
                lam = g[j] if j >= 0 else -np.inf
                _record(summ, tests, rep, inp, delta, alpha, 'auto_sel', lam, first_stop(TEx, sc['te'], [lam]), TEx, delta, choice=best)
                if light:
                    continue
                msk = CALx.ID_test.isin(one_pp).values   # one calibration test per participant
                j = ltt_select_ordered(CALx[msk], sc['cal'][msk], g, delta, alpha, ETA, order, selective=True)
                lam = g[j] if j >= 0 else -np.inf
                _record(summ, tests, rep, inp, delta, alpha, 'auto_sel_cal1pp', lam, first_stop(TEx, sc['te'], [lam]), TEx, delta, choice=best)
                _record(summ, tests, rep, inp, delta, alpha, 'auto_sel_bysex', np.nan,
                        first_stop(TEx, bysex_scores(CALx, TEx, sc['cal'], sc['te'], g, gcol, delta, alpha, selective=True, order=order), [0.0]), TEx, delta, choice=best)
            if not light:
                # uncertified rules: interval inside +-delta, split-conformal interval inside +-delta, 85 % HRmax end point
                # with the anytime GBM and with the clinical extrapolation estimator; privileged oracle
                _record(summ, tests, rep, inp, delta, np.nan, 'width_naive', 2 * delta, first_stop(TEx, gate(TEx.width.values, 'te'), [2 * delta]), TEx, delta)
                ce = el['cal']
                E = np.maximum(CALx.qlo.values - CALx.y.values, CALx.y.values - CALx.qhi.values)[ce]
                Q = np.quantile(E, min(1.0, np.ceil((len(E) + 1) * 0.9) / len(E)))
                _record(summ, tests, rep, inp, delta, np.nan, 'conformal90', delta, first_stop(TEx, gate((TEx.width.values + 2 * Q) / 2, 'te'), [delta]), TEx, delta)
                _record(summ, tests, rep, inp, delta, np.nan, 'hr85_uncal', -0.85, first_stop(TEx, -TEx.hr_frac.values, [-0.85]), TEx, delta)
                clin = pr.get('hr_extrap') if dt else pr.get('acsm_extrap')
                if clin is not None:
                    TEc = TEx.copy(); TEc['err'] = clin - TEc.y.values
                    _record(summ, tests, rep, inp, delta, np.nan, 'hr85_clinical', -0.85, first_stop(TEc, -TEc.hr_frac.values, [-0.85]), TEc, delta)
                ok = (TEx.err.abs() <= delta).values
                settled = pd.Series(ok[::-1]).groupby(TEx.ID_test.values[::-1], sort=False).cummin().values[::-1]
                _record(summ, tests, rep, inp, delta, np.nan, 'oracle', 0.5, first_stop(TEx, gate(np.where(settled, 0.0, 1.0), 'te'), [0.5]), TEx, delta)
    pd.DataFrame(summ).to_csv(f'{outdir}/parts/summ_rep{rep:02d}.csv', index=False)
    pd.concat(tests).to_parquet(f'{outdir}/parts/tests_rep{rep:02d}.parquet')
    pd.concat(preds).to_parquet(f'{outdir}/parts/preds_rep{rep:02d}.parquet')
    pd.DataFrame(timing).to_csv(f'{outdir}/parts/timing_rep{rep:02d}.csv', index=False)
    pd.DataFrame(calib).to_csv(f'{outdir}/parts/calib_rep{rep:02d}.csv', index=False)
    return rep, time.time() - t0


def bysex_scores(CALx, TEx, s_cal, s_te, grid, gcol, delta, alpha, selective, order=None):
    """Score shifted by a threshold certified separately for each sex; stop when the shifted score is <= 0."""
    lam_g = {}
    for gv in TEx[gcol].dropna().unique():
        mc = (CALx[gcol] == gv).values
        if not mc.sum():
            lam_g[gv] = -np.inf; continue
        if selective:
            j = ltt_select_ordered(CALx[mc], s_cal[mc], grid, delta, alpha, ETA, order, selective=True)
        else:
            j = ltt_select(CALx[mc], s_cal[mc], grid, delta, alpha, ETA, selective=False)
        lam_g[gv] = grid[j] if j >= 0 else -np.inf
    lam_row = TEx[gcol].map(lam_g).fillna(-np.inf).values
    return np.where(np.isfinite(lam_row), s_te - np.where(np.isfinite(lam_row), lam_row, 0.0), np.inf)


def first_eligible_random(D, el, rng):
    """No-information score: a uniform random number at the first eligible decision of each test, +inf elsewhere."""
    s = np.full(len(D), np.inf)
    df = pd.DataFrame({'id': D.ID_test.values, 'el': el, 'i': np.arange(len(D))})
    fe = df[df.el].groupby('id', sort=False).i.first()
    s[fe.values] = rng.random(len(fe))
    return s


def _auc(p, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def _record(summ, tests, rep, inp, delta, alpha, rule, lam, o, TEx, d, choice=None):
    st = o['stopped'][:, 0]; err = o['err'][:, 0]; bad = (np.abs(err) > d) & st
    ids = TEx.ID_test.drop_duplicates().values
    fe = TEx[TEx.hr_frac >= GATE].groupby('ID_test', sort=False).tsec.first().reindex(ids).values
    tau = o['tau'][:, 0]
    first = st & np.isclose(tau, fe)
    summ.append(dict(rep=rep, input=inp, delta=delta, alpha=alpha, rule=rule, lam=lam, n=len(st), risk=bad.mean(),
                     sel_risk=bad.sum() / st.sum() if st.sum() else np.nan, n_stopped=int(st.sum()), n_bad=int(bad.sum()), early=st.mean(), saved_s=o['saved'][:, 0].mean(),
                     saved_frac=1 - o['frac'][:, 0].mean(), hrfrac=o['hrfrac'][:, 0].mean(),
                     below90=(st & (o['hrfrac'][:, 0] < 0.90)).mean(), mae_stopped=np.abs(err[st]).mean() if st.any() else np.nan,
                     share_first=first.sum() / max(st.sum(), 1), median_frac_stopped=np.median(o['frac'][st, 0]) if st.any() else np.nan, choice=choice))
    tests.append(pd.DataFrame(dict(rep=rep, input=inp, delta=delta, alpha=alpha, rule=rule, ID_test=ids, stopped=st, err=err,
                                   saved_s=o['saved'][:, 0], frac=o['frac'][:, 0], hrfrac=o['hrfrac'][:, 0], tau=tau, first_elig=first)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True, choices=['malaga', 'dublin'])
    ap.add_argument('--reps', type=int, default=20); ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()
    P, FEATS = load(a.ds)
    suffix = '' if abs(GATE - 0.77) < 1e-9 and not LIGHT else f'_gate{int(round(GATE * 100))}'
    outdir, wdir = f'results/{a.ds}{suffix}', f'weights/{a.ds}{suffix}'
    os.makedirs(f'{outdir}/parts', exist_ok=True); os.makedirs(wdir, exist_ok=True)
    json.dump({k: v for k, v in FEATS.items()}, open(f'{outdir}/features.json', 'w'), indent=1)
    print(a.ds, P.shape, P.ID_test.nunique(), 'tests', P.ID.nunique(), 'subjects', {k: len(v) for k, v in FEATS.items()}, flush=True)
    reps = [r for r in range(a.start, a.start + a.reps) if not os.path.exists(f'{outdir}/parts/calib_rep{r:02d}.csv')]
    from joblib import Parallel, delayed
    for rep, dt_ in Parallel(n_jobs=a.jobs, verbose=5)(delayed(run_rep)(a.ds, r, P, FEATS, outdir, wdir) for r in reps):
        print(f'rep {rep} done in {dt_:.0f}s', flush=True)
    import glob
    for pat, out in (('summ', 'stop_summary.csv'), ('calib', 'cls_calib.csv'), ('timing', 'timing.csv')):
        pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{outdir}/parts/{pat}_rep*.csv'))]).to_csv(f'{outdir}/{out}', index=False)
    for pat, out in (('tests', 'stop_tests.parquet'), ('preds', 'pred_rows.parquet')):
        pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{outdir}/parts/{pat}_rep*.parquet'))]).to_parquet(f'{outdir}/{out}')
    print('all done')


if __name__ == '__main__':
    main()
