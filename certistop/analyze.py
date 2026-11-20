"""Aggregate the repeated-split results.

Inference over the 20 overlapping random splits uses the corrected resampled t-test of Nadeau and Bengio (2003):
the variance of a mean over J splits is inflated by (1/J + n_test/n_rest), with n_test/n_rest = 0.25/0.75.
Families for Holm's correction are declared per table (all comparisons that appear in the same table).
Subgroup risks use a participant-level bootstrap over pooled out-of-sample outcomes.
Writes results/summary/*.csv."""
import os, glob
import numpy as np, pandas as pd
from scipy import stats

OUT = 'results/summary'
os.makedirs(OUT, exist_ok=True)
RATIO = 0.25 / 0.75
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)


def cse(x):
    """Corrected standard error of the mean of per-split values."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]; J = len(x)
    return np.sqrt((1.0 / J + RATIO) * x.var(ddof=1)) if J > 1 else np.nan


def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return stats.t.ppf(0.975, len(x) - 1) * cse(x) if len(x) > 1 else np.nan


def ctest(d):
    """Two-sided corrected resampled paired t-test on per-split differences."""
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if len(d) < 2 or np.allclose(d, 0):
        return 1.0
    t = d.mean() / cse(d)
    return 2 * stats.t.sf(abs(t), len(d) - 1)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj


def load_preds(ds):
    P = pd.read_parquet(f'results/{ds}/pred_rows.parquet')
    for f in glob.glob(f'results/{ds}/seqnn_*.parquet'):
        P = pd.concat([P, pd.read_parquet(f)])
    P['ae'] = (P.yhat - P.y).abs()
    return P.sort_values(['rep', 'input', 'model', 'ID_test', 'tsec'])


def predictor_tables(ds):
    P = load_preds(ds)
    last = P.groupby(['rep', 'input', 'model', 'ID_test']).tail(1)            # true last decision before exhaustion
    s85 = P[P.hr_frac >= 0.85].groupby(['rep', 'input', 'model', 'ID_test']).head(1)
    rows = []
    for (inp, mdl), g in P.groupby(['input', 'model']):
        per = g.groupby('rep').ae.mean()
        pl = last[(last.input == inp) & (last.model == mdl)].groupby('rep').ae.mean()
        p85 = s85[(s85.input == inp) & (s85.model == mdl)].groupby('rep').ae.mean()
        rows.append(dict(ds=ds, input=inp, model=mdl, mae=per.mean(), mae_ci=ci(per), mae_last=pl.mean(), mae_last_ci=ci(pl),
                         mae_hr85=p85.mean(), mae_hr85_ci=ci(p85), rmse=np.sqrt(((g.yhat - g.y) ** 2).mean())))
    T = pd.DataFrame(rows)
    tests = []
    for nm, D in (('all', P), ('last', last)):
        piv = D.groupby(['input', 'model', 'rep']).ae.mean().unstack('model')
        for inp in piv.index.get_level_values(0).unique():
            x = piv.loc[inp]
            for m in x.columns:
                if m != 'main' and x[m].notna().all():
                    d = x['main'] - x[m]
                    tests.append(dict(ds=ds, point=nm, input=inp, model=m, diff=d.mean(), diff_ci=ci(d), p=ctest(d)))
    return T, pd.DataFrame(tests)


def stop_tables(ds):
    S = pd.read_csv(f'results/{ds}/stop_summary.csv')
    S['alpha'] = S.alpha.fillna(-1)
    S['ok'] = np.where(S.alpha > 0, (S.risk <= S.alpha).astype(float), np.nan)
    S['ok_sel'] = np.where(S.alpha > 0, (S.sel_risk <= S.alpha).astype(float), np.nan)
    agg = S.groupby(['input', 'delta', 'alpha', 'rule']).agg(
        risk=('risk', 'mean'), risk_ci=('risk', ci), sel_risk=('sel_risk', 'mean'), sel_risk_ci=('sel_risk', ci),
        early=('early', 'mean'), saved_s=('saved_s', 'mean'), saved_ci=('saved_s', ci), saved_frac=('saved_frac', 'mean'),
        hrfrac=('hrfrac', 'mean'), below90=('below90', 'mean'), mae_stopped=('mae_stopped', 'mean'),
        share_first=('share_first', 'mean'), median_frac=('median_frac_stopped', 'mean'),
        valid=('ok', 'mean'), valid_sel=('ok_sel', 'mean'), n=('rep', 'size')).reset_index()
    tests = []
    for ref in ('auto', 'auto_sel', 'width', 'width_sel'):
        for (inp, d, a), g in S[S.alpha > 0].groupby(['input', 'delta', 'alpha']):
            piv = g.pivot(index='rep', columns='rule', values='saved_s')
            if ref not in piv:
                continue
            for c in piv.columns:
                if c == ref or ref.endswith('_sel') != c.endswith('_sel') or c in ('auto', 'auto_sel') and ref.startswith('width'):
                    continue
                dlt = piv[ref] - piv[c]
                tests.append(dict(ds=ds, ref=ref, input=inp, delta=d, alpha=a, vs=c, diff=dlt.mean(), diff_ci=ci(dlt), wins=(dlt > 0).mean(), p=ctest(dlt)))
    return agg, pd.DataFrame(tests)


def subgroup_table(ds, delta=5.0, alpha=0.10, rule='width', B=2000, seed=0):
    """Participant-level bootstrap of subgroup risk from pooled out-of-sample outcomes (each test is in the test split
    of about 5 of the 20 splits; its loss is averaged over those splits)."""
    T = pd.read_parquet(f'results/{ds}/stop_tests.parquet')
    T = T[(T.rule == rule) & (T.delta == delta) & (T.alpha == alpha)].copy()
    info = pd.read_csv(f'data/{ds}_tests.csv')
    if ds == 'malaga':
        si = pd.read_csv('data_raw/malaga/subject-info.csv')[['ID_test', 'Age', 'Sex']].rename(columns={'Age': 'age', 'Sex': 'sex'})
        si['sex'] = si.sex.map({0: 'M', 1: 'F'})
        info = info.drop(columns=[c for c in ('sex', 'age') if c in info]).merge(si, on='ID_test')
    T = T.merge(info[['ID_test', 'ID', 'sex', 'age'] + (['sport'] if 'sport' in info else [])], on='ID_test')
    T['age_grp'] = pd.cut(T.age, [0, 18, 35, 120], labels=['<18', '18-35', '>35'], right=False)
    T['bad'] = (T.stopped & (T.err.abs() > delta)).astype(float)
    T['stp'] = T.stopped.astype(float)
    rng = np.random.default_rng(seed)
    rows = []
    for col in ['sex', 'age_grp'] + (['sport'] if 'sport' in T else []):
        for (inp, lev), g in T.groupby(['input', col], observed=True):
            pt = g.groupby('ID_test').agg(bad=('bad', 'mean'), saved=('saved_s', 'mean'), stopped=('stp', 'mean'), ID=('ID', 'first'))
            pp = pt.groupby('ID').agg(bad=('bad', 'sum'), n=('bad', 'size'), stp=('stopped', 'sum'))
            ids = pp.index.values; est = pp.bad.sum() / pp.n.sum()
            sel = pp.bad.sum() / max(pp.stp.sum(), 1e-9)
            bs, bss = [], []
            for _ in range(B):
                s = pp.loc[rng.choice(ids, len(ids), replace=True)]
                bs.append(s.bad.sum() / s.n.sum()); bss.append(s.bad.sum() / max(s.stp.sum(), 1e-9))
            bs, bss = np.array(bs), np.array(bss)
            rows.append(dict(ds=ds, rule=rule, alpha=alpha, input=inp, group=col, level=str(lev), n_tests=len(pt), n_participants=len(pp),
                             risk=est, lo=np.quantile(bs, 0.025), hi=np.quantile(bs, 0.975), p_exceed=(bs <= alpha).mean(),
                             sel_risk=sel, sel_lo=np.quantile(bss, 0.025), sel_hi=np.quantile(bss, 0.975),
                             saved_s=pt.saved.mean(), early=pt.stopped.mean()))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    allW = []
    for ds in ('malaga', 'dublin'):
        if not os.path.exists(f'results/{ds}/stop_summary.csv'):
            continue
        T, PW = predictor_tables(ds); T.to_csv(f'{OUT}/{ds}_predictors.csv', index=False); PW.to_csv(f'{OUT}/{ds}_predictor_tests.csv', index=False)
        print('=====', ds, 'predictors'); print(T.round(2).to_string()); print(PW.round(4).to_string())
        A, W = stop_tables(ds); A.to_csv(f'{OUT}/{ds}_stop.csv', index=False); allW.append(W)
        C = pd.read_csv(f'results/{ds}/cls_calib.csv')
        C.groupby(['input', 'delta'])[['brier', 'ece', 'auc', 'auc_width', 'base_rate']].mean().to_csv(f'{OUT}/{ds}_riskclf_calib.csv')
        print(C.groupby(['input', 'delta'])[['brier', 'ece', 'auc', 'auc_width', 'base_rate']].mean().round(3))
        Tm = pd.read_csv(f'results/{ds}/timing.csv'); Tm.groupby('input').mean(numeric_only=True).to_csv(f'{OUT}/{ds}_timing.csv')
        print(Tm.groupby('input').mean(numeric_only=True).round(4))
        G = pd.concat([subgroup_table(ds, rule=r) for r in ('width', 'width_bysex')] +
                      [subgroup_table(ds, rule='width_sel', alpha=a) for a in (0.15, 0.20)])
        G.to_csv(f'{OUT}/{ds}_subgroups.csv', index=False)
    W = pd.concat(allW).reset_index(drop=True)
    W['p_holm'] = np.nan
    # Holm family: all comparisons of one reference rule at one risk level (both datasets, inputs and tolerances)
    for _, idx in W.groupby(['ref', 'alpha']).groups.items():
        W.loc[idx, 'p_holm'] = holm(W.loc[idx, 'p'].values)
    W.to_csv(f'{OUT}/stop_tests_all.csv', index=False)
