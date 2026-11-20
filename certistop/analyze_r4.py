"""Round-4 analyses: (1) permutation null with per-split paired differences and corrected resampled tests,
(2) comparison with the 85 % HRmax end point at matched share of stopped tests and at matched time saved,
(3) risk-coverage curves interpolated on a fine coverage grid. Writes results/summary/r4_*.csv."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import ci, ctest, cse

OUT = 'results/summary'
GATE = 0.77
pd.set_option('display.width', 250)


def per_split_permutation(ds, alphas=(0.10, 0.15, 0.20), delta=5.0, B=50, seed=0):
    P = pd.read_parquet(f'results/{ds}/pred_rows.parquet'); P = P[(P.input == 'full') & (P.model == 'main')].sort_values(['rep', 'ID_test', 'tsec'])
    T = pd.read_parquet(f'results/{ds}/stop_tests.parquet'); T = T[(T.input == 'full') & (T.rule == 'auto_sel') & (T.delta == delta)]
    rng = np.random.default_rng(seed); rows = []
    for rep, Pr in P.groupby('rep'):
        E = Pr[Pr.hr_frac >= GATE]
        seqs = {t: (g.yhat.values - g.y.values, g.tsec.values, g.Tsec.iloc[0]) for t, g in E.groupby('ID_test')}
        for a in alphas:
            x = T[(T.rep == rep) & (T.alpha == a)]
            ids = x.ID_test.values
            k = np.array([int(np.searchsorted(seqs[t][1], tau)) if st and t in seqs else -1 for t, st, tau in zip(ids, x.stopped.values, x.tau.values)])
            n_c = int(x.stopped.sum()); bad_c = int((x.stopped & (x.err.abs() > delta)).sum())
            if n_c == 0:
                continue
            errs = []
            for b in range(B):
                kp = rng.permutation(k); n = bad = 0
                for t, kk in zip(ids, kp):
                    if kk < 0 or t not in seqs or kk >= len(seqs[t][0]):
                        continue
                    n += 1; bad += abs(seqs[t][0][kk]) > delta
                if n:
                    errs.append(bad / n)
            rows.append(dict(ds=ds, rep=rep, alpha=a, err_certistop=bad_c / n_c, err_perm=np.mean(errs), gap=np.mean(errs) - bad_c / n_c))
    R = pd.DataFrame(rows)
    S = R.groupby(['ds', 'alpha']).agg(splits=('gap', 'size'), err_certistop=('err_certistop', 'mean'), err_perm=('err_perm', 'mean'),
                                         gap=('gap', 'mean'), gap_ci=('gap', ci), p=('gap', ctest)).reset_index()
    return R, S


def matched_85(ds, delta=5.0):
    """Error among stopped tests of each score at the share stopped (and at the time saved) of the 85 % HRmax end point,
    using the per-split threshold curves of the test tests (descriptive: thresholds are read off the test tests)."""
    St = pd.read_csv(f'results/{ds}/stop_summary.csv')
    h = St[(St.input == 'full') & (St.delta == delta) & (St.rule == 'hr85_uncal')].set_index('rep')
    R = pd.read_parquet(f'results/{ds}/riskcov.parquet'); R = R[(R.input == 'full') & (R.delta == delta)]
    rows = []
    for (rep, sc), g in R.groupby(['rep', 'score']):
        g = g.sort_values('coverage'); cov85 = h.loc[rep, 'early']; sav85 = h.loc[rep, 'saved_s']
        jc = (g.coverage - cov85).abs().idxmin(); js = (g.saved_s - sav85).abs().idxmin()
        rows.append(dict(ds=ds, rep=rep, score=sc, cov85=cov85, err85=h.loc[rep, 'n_bad'] / h.loc[rep, 'n_stopped'],
                         cov_at=g.loc[jc, 'coverage'], err_at_cov=g.loc[jc, 'sel_risk'], saved_at_cov=g.loc[jc, 'saved_s'],
                         err_at_saved=g.loc[js, 'sel_risk'], cov_at_saved=g.loc[js, 'coverage']))
    M = pd.DataFrame(rows)
    return M.groupby(['ds', 'score']).agg(cov85=('cov85', 'mean'), err85=('err85', 'mean'), cov_at=('cov_at', 'mean'), err_at_cov=('err_at_cov', 'mean'),
                                          saved_at_cov=('saved_at_cov', 'mean'), err_at_saved=('err_at_saved', 'mean'), cov_at_saved=('cov_at_saved', 'mean')).reset_index()


def interp_curves(ds, delta=5.0, grid=np.round(np.arange(0.05, 0.991, 0.01), 2)):
    R = pd.read_parquet(f'results/{ds}/riskcov.parquet'); R = R[(R.input == 'full') & (R.delta == delta)]
    rows = []
    for (rep, sc), g in R.groupby(['rep', 'score']):
        g = g.dropna(subset=['sel_risk']).sort_values('coverage').drop_duplicates('coverage')
        if len(g) < 2:
            continue
        v = np.interp(grid, g.coverage.values, g.sel_risk.values, left=np.nan, right=np.nan)
        rows += [dict(ds=ds, rep=rep, score=sc, coverage=c, sel_risk=e) for c, e in zip(grid, v)]
    C = pd.DataFrame(rows)
    return C.groupby(['ds', 'score', 'coverage']).sel_risk.mean().reset_index()


if __name__ == '__main__':
    Rs, Ss = zip(*[per_split_permutation(ds) for ds in ('malaga', 'dublin')])
    pd.concat(Rs).to_csv(f'{OUT}/r4_permutation_splits.csv', index=False); S = pd.concat(Ss); S.to_csv(f'{OUT}/r4_permutation_tests.csv', index=False)
    print(S.round(4).to_string(index=False))
    M = pd.concat([matched_85(ds) for ds in ('malaga', 'dublin')]); M.to_csv(f'{OUT}/r4_matched85.csv', index=False)
    print(M.round(3).to_string(index=False))
    C = pd.concat([interp_curves(ds) for ds in ('malaga', 'dublin')]); C.to_csv(f'{OUT}/r4_riskcov_interp.csv', index=False)
    print(C[C.coverage.isin([0.5, 0.8, 0.9, 0.95, 0.97, 0.99])].pivot_table(index=['ds', 'coverage'], columns='score', values='sel_risk').round(3).to_string())
