"""Round-5 analyses: (1) permutation null for CertiStop and for fixed per-stop scores (interval width, lactate, timing only),
with the share of tests stopped by each rule and by its null, and Holm correction over the six CertiStop tests,
(2) error among stopped tests of the 85 % HRmax end point and of the w <= 2 delta rule, pooled over splits,
(3) error of each score at the time saved of the 85 % end point, interpolated on each split's threshold curve.
Writes results/summary/r5_*.csv."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import ci, ctest, holm

OUT = 'results/summary'
GATE = 0.77
pd.set_option('display.width', 250)


def permutation(ds, rule, alphas=(0.10, 0.15, 0.20), delta=5.0, B=50, seed=0):
    P = pd.read_parquet(f'results/{ds}/pred_rows.parquet'); P = P[(P.input == 'full') & (P.model == 'main')].sort_values(['rep', 'ID_test', 'tsec'])
    T = pd.read_parquet(f'results/{ds}/stop_tests.parquet'); T = T[(T.input == 'full') & (T.rule == rule) & (T.delta == delta)]
    rng = np.random.default_rng(seed); rows = []
    for rep, Pr in P.groupby('rep'):
        E = Pr[Pr.hr_frac >= GATE]
        seqs = {t: (g.yhat.values - g.y.values, g.tsec.values) for t, g in E.groupby('ID_test')}
        for a in alphas:
            x = T[(T.rep == rep) & (T.alpha == a)]
            ids = x.ID_test.values
            k = np.array([int(np.searchsorted(seqs[t][1], tau)) if st and t in seqs else -1 for t, st, tau in zip(ids, x.stopped.values, x.tau.values)])
            n_c = int(x.stopped.sum()); bad_c = int((x.stopped & (x.err.abs() > delta)).sum())
            if n_c == 0:
                continue
            errs, shares = [], []
            for b in range(B):
                kp = rng.permutation(k); n = bad = 0
                for t, kk in zip(ids, kp):
                    if kk < 0 or t not in seqs or kk >= len(seqs[t][0]):
                        continue
                    n += 1; bad += abs(seqs[t][0][kk]) > delta
                shares.append(n / len(ids))
                if n:
                    errs.append(bad / n)
            rows.append(dict(ds=ds, rule=rule, rep=rep, alpha=a, err_rule=bad_c / n_c, err_perm=np.mean(errs), gap=np.mean(errs) - bad_c / n_c,
                             share_rule=n_c / len(ids), share_perm=np.mean(shares)))
    return pd.DataFrame(rows)


def end_point_errors(ds, delta=5.0):
    S = pd.read_csv(f'results/{ds}/stop_summary.csv'); S = S[(S.delta == delta) & (S.alpha.isna() | (S.alpha < 0) | S.rule.isin(['hr85_uncal', 'hr85_clinical', 'width_naive', 'conformal90']))]
    rows = []
    for (inp, rule), g in S[S.rule.isin(['hr85_uncal', 'hr85_clinical', 'width_naive', 'conformal90'])].groupby(['input', 'rule']):
        g = g.drop_duplicates('rep')
        rows.append(dict(ds=ds, input=inp, rule=rule, budget_risk=g.risk.mean(), err_stopped_pooled=g.n_bad.sum() / max(g.n_stopped.sum(), 1),
                         share_stopped=g.early.mean(), saved_s=g.saved_s.mean()))
    return pd.DataFrame(rows)


def matched_saving(ds, delta=5.0):
    St = pd.read_csv(f'results/{ds}/stop_summary.csv')
    h = St[(St.input == 'full') & (St.delta == delta) & (St.rule == 'hr85_uncal')].drop_duplicates('rep').set_index('rep')
    R = pd.read_parquet(f'results/{ds}/riskcov.parquet'); R = R[(R.input == 'full') & (R.delta == delta)]
    rows = []
    for (rep, sc), g in R.groupby(['rep', 'score']):
        g = g.dropna(subset=['sel_risk']).sort_values('saved_s').drop_duplicates('saved_s')
        s85 = h.loc[rep, 'saved_s']
        if len(g) < 2 or s85 > g.saved_s.max():
            continue
        rows.append(dict(ds=ds, rep=rep, score=sc, saved85=s85, err85=h.loc[rep, 'n_bad'] / h.loc[rep, 'n_stopped'],
                         err_at_saved=np.interp(s85, g.saved_s.values, g.sel_risk.values)))
    M = pd.DataFrame(rows)
    return M.groupby(['ds', 'score']).agg(splits=('rep', 'size'), err85=('err85', 'mean'), err_at_saved=('err_at_saved', 'mean')).reset_index()


if __name__ == '__main__':
    R = pd.concat([permutation('malaga', r) for r in ('auto_sel', 'width_sel', 'since_gate_sel')] +
                  [permutation('dublin', r) for r in ('auto_sel', 'lactate_sel', 'since_gate_sel')])
    R.to_csv(f'{OUT}/r5_permutation_splits.csv', index=False)
    S = R.groupby(['ds', 'rule', 'alpha']).agg(splits=('gap', 'size'), err_rule=('err_rule', 'mean'), err_perm=('err_perm', 'mean'), gap=('gap', 'mean'),
                                               gap_ci=('gap', ci), p=('gap', ctest), share_rule=('share_rule', 'mean'), share_perm=('share_perm', 'mean')).reset_index()
    S['p_holm_certistop'] = np.nan
    m = S.rule == 'auto_sel'
    S.loc[m, 'p_holm_certistop'] = holm(S.loc[m, 'p'].values)
    S.to_csv(f'{OUT}/r5_permutation_tests.csv', index=False)
    print(S.round(4).to_string(index=False))
    E = pd.concat([end_point_errors(ds) for ds in ('malaga', 'dublin')]); E.to_csv(f'{OUT}/r5_uncertified.csv', index=False)
    print(E.round(4).to_string(index=False))
    M = pd.concat([matched_saving(ds) for ds in ('malaga', 'dublin')]); M.to_csv(f'{OUT}/r5_matched_saving.csv', index=False)
    print(M.round(4).to_string(index=False))
