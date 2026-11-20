"""Additional analyses: pooled error among stopped tests, permutation null at matched timing, Dublin half-stage
sensitivity, HR at stop among stopped tests, per-stop subgroup audit, gate sensitivity and matched-coverage comparison.
Writes results/summary/r3_*.csv."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import ci, ctest, holm, subgroup_table

OUT = 'results/summary'
GATE = 0.77
pd.set_option('display.width', 250)


def pooled_table(ds):
    """Error among stopped tests pooled over splits, mean over splits with stops, splits with stops, HR at stop."""
    S = pd.read_csv(f'results/{ds}/stop_summary.csv'); T = pd.read_parquet(f'results/{ds}/stop_tests.parquet')
    S['alpha'] = S.alpha.fillna(-1); T['alpha'] = T.alpha.fillna(-1)
    g = S.groupby(['input', 'delta', 'alpha', 'rule'])
    A = g.agg(risk=('risk', 'mean'), saved_s=('saved_s', 'mean'), saved_ci=('saved_s', ci), early=('early', 'mean'),
              n_stop=('n_stopped', 'sum'), n_bad=('n_bad', 'sum'), splits_with_stops=('n_stopped', lambda s: int((s > 0).sum())),
              sel_mean_nonempty=('sel_risk', 'mean'), median_frac=('median_frac_stopped', 'mean')).reset_index()
    A['sel_pooled'] = np.where(A.n_stop > 0, A.n_bad / A.n_stop.clip(lower=1), np.nan)
    hs = T[T.stopped].groupby(['input', 'delta', 'alpha', 'rule']).hrfrac.mean().rename('hr_at_stop')
    b90 = T.groupby(['input', 'delta', 'alpha', 'rule']).apply(lambda x: (x.stopped & (x.hrfrac < 0.9)).mean()).rename('below90')
    A = A.merge(hs.reset_index(), how='left').merge(b90.reset_index(), how='left')
    if ds == 'dublin':     # half-stage sensitivity: the exhaustion stage completed only half on average
        info = pd.read_csv('data/dublin_tests.csv')[['ID_test', 'T', 'Tsec']]; info['stage_len'] = info.Tsec / info['T']
        T2 = T.merge(info[['ID_test', 'stage_len']], on='ID_test')
        T2['saved_half'] = np.where(T2.stopped, np.maximum(T2.saved_s - 0.5 * T2.stage_len, 0.0), 0.0)
        sh = T2.groupby(['input', 'delta', 'alpha', 'rule', 'rep']).saved_half.mean().groupby(['input', 'delta', 'alpha', 'rule']).mean().rename('saved_half_stage')
        A = A.merge(sh.reset_index(), how='left')
    A['ds'] = ds
    return A


def permutation_null(ds, rules=('auto_sel',), alphas=(0.10, 0.15, 0.20), deltas=(5.0,), B=20, seed=0):
    """Give each test the stop index (counted in eligible decisions from the gate) of another test of the same split;
    keeps the timing distribution of the rule and removes person-specific information."""
    P = pd.read_parquet(f'results/{ds}/pred_rows.parquet'); P = P[(P.input == 'full') & (P.model == 'main')].sort_values(['rep', 'ID_test', 'tsec'])
    P['el'] = P.hr_frac >= GATE
    T = pd.read_parquet(f'results/{ds}/stop_tests.parquet'); T = T[T.input == 'full']
    rng = np.random.default_rng(seed); rows = []
    for rep, Pr in P.groupby('rep'):
        E = Pr[Pr.el].copy(); E['k'] = E.groupby('ID_test').cumcount()
        seqs = {t: (g.yhat.values - g.y.values, g.tsec.values, g.Tsec.iloc[0]) for t, g in E.groupby('ID_test')}
        Tsec = Pr.groupby('ID_test').Tsec.first()
        for rule in rules:
            for a in alphas:
                for d in deltas:
                    x = T[(T.rep == rep) & (T.rule == rule) & (T.alpha == a) & (T.delta == d)]
                    if not len(x):
                        continue
                    ids = x.ID_test.values; kstop = []
                    for t, st, tau in zip(ids, x.stopped.values, x.tau.values):
                        if not st or t not in seqs:
                            kstop.append(-1); continue
                        kstop.append(int(np.searchsorted(seqs[t][1], tau)))
                    kstop = np.array(kstop)
                    for b in range(B):
                        kp = rng.permutation(kstop); bad = n = 0; saved = 0.0
                        for t, k in zip(ids, kp):
                            if k < 0 or t not in seqs or k >= len(seqs[t][0]):
                                continue
                            e, ts, Ts = seqs[t]; n += 1; bad += abs(e[k]) > d; saved += Ts - ts[k]
                        rows.append(dict(ds=ds, rep=rep, rule=rule, alpha=a, delta=d, b=b, n_stop=n, n_bad=bad, saved_s=saved / len(ids)))
    R = pd.DataFrame(rows)
    return R.groupby(['ds', 'rule', 'alpha', 'delta']).agg(n_stop=('n_stop', 'sum'), n_bad=('n_bad', 'sum'), saved_s=('saved_s', 'mean')).assign(
        sel_pooled=lambda z: z.n_bad / z.n_stop).reset_index()


if __name__ == '__main__':
    A = pd.concat([pooled_table(ds) for ds in ('malaga', 'dublin')]); A.to_csv(f'{OUT}/r3_pooled.csv', index=False)
    key = A[(A.input == 'full') & (A.delta == 5.0) & A.rule.str.endswith('_sel') | (A.rule == 'auto_sel')]
    print(key[(key.input == 'full') & (key.delta == 5.0)][['ds', 'alpha', 'rule', 'saved_s', 'saved_ci', 'sel_pooled', 'splits_with_stops', 'early', 'hr_at_stop', 'below90'] +
          (['saved_half_stage'] if 'saved_half_stage' in key else [])].round(3).to_string(index=False))
    N = pd.concat([permutation_null(ds) for ds in ('malaga', 'dublin')]); N.to_csv(f'{OUT}/r3_permutation.csv', index=False); print(N.round(3).to_string(index=False))
    # per-stop subgroup audit of CertiStop and of per-sex certification
    G = []
    for ds in ('malaga', 'dublin'):
        for rule in ('auto_sel', 'auto_sel_bysex'):
            for a in (0.10, 0.15, 0.20):
                G.append(subgroup_table(ds, rule=rule, alpha=a))
    G = pd.concat(G); G.to_csv(f'{OUT}/r3_subgroups_sel.csv', index=False)
    ex = G[(G.input == 'full') & (G.rule == 'auto_sel') & (G.sel_risk > G.alpha)]
    print('per-stop exceedances (lab):'); print(ex[['ds', 'alpha', 'group', 'level', 'n_tests', 'sel_risk', 'sel_lo', 'sel_hi']].round(3).to_string(index=False))
    # tests vs CertiStop under the per-stop guarantee, all alpha, Holm over the per-stop table family
    W = pd.read_csv(f'{OUT}/stop_tests_all.csv')
    F = W[(W.ref == 'auto_sel') & (W.input == 'full') & (W.delta == 5.0)].copy()
    F['p_holm_table'] = holm(F.p.values); F.to_csv(f'{OUT}/r3_sel_tests.csv', index=False)
    print(F[['ds', 'alpha', 'vs', 'diff', 'diff_ci', 'p', 'p_holm_table']].round(4).to_string(index=False))
