"""Core of the risk-controlled early-termination framework: stopping evaluation and Learn-then-Test calibration."""
import numpy as np, pandas as pd
from scipy.stats import binom

def first_stop(D, s, lams):
    """D: prefix rows sorted by (ID_test, tsec) with columns ID_test, tsec, Tsec, err, hr, hrmax.
    s: stop score per row (stop at first row with s <= lam). Returns dict of arrays (n_tests, n_lams)."""
    D = D.assign(_s=s)
    lams = np.asarray(lams, float)
    out = {k: [] for k in ('frac', 'err', 'stopped', 'saved', 'hrfrac', 'tau')}
    for _, g in D.groupby('ID_test', sort=False):
        rm = np.minimum.accumulate(g._s.values)
        idx = np.searchsorted(-rm, -lams, side='left')
        n = len(g); T = g['Tsec'].iloc[0]; hm = g.hrmax.iloc[0]
        st = idx < n; ii = np.minimum(idx, n - 1)
        tau = np.where(st, g.tsec.values[ii], T)
        out['tau'].append(tau); out['frac'].append(tau / T); out['saved'].append(T - tau)
        out['err'].append(np.where(st, g.err.values[ii], 0.0)); out['stopped'].append(st)
        out['hrfrac'].append(np.where(st, g.hr.values[ii], hm) / hm)
    return {k: np.array(v) for k, v in out.items()}

def risk_counts(o, delta, selective):
    bad = (np.abs(o['err']) > delta) & o['stopped']
    k = bad.sum(0); n = o['stopped'].sum(0) if selective else np.full(bad.shape[1], bad.shape[0])
    return k, n

def ltt_select(D, s, lams, delta, alpha, eta=0.1, selective=True, start=0):
    """Fixed-sequence Learn-then-Test over lams ordered conservative->aggressive, starting at index `start`
    (chosen without calibration labels). H_lam: risk(lam) > alpha, binomial p-value. Returns index of the most
    aggressive certified lam, or -1."""
    o = first_stop(D, s, lams)
    k, n = risk_counts(o, delta, selective)
    best = -1
    for j in range(start, len(lams)):
        p = binom.cdf(k[j], n[j], alpha) if n[j] > 0 else 1.0
        if p <= eta: best = j
        else: break
    return best


def ltt_select_ordered(D, s, lams, delta, alpha, eta, order, selective=True):
    """Fixed-sequence Learn-then-Test along a pre-specified order of grid indices (e.g. ordered by p-values computed
    on training data, never on calibration labels). Returns the most aggressive certified index, or -1."""
    o = first_stop(D, s, lams)
    k, n = risk_counts(o, delta, selective)
    acc = []
    for j in order:
        p = binom.cdf(k[j], n[j], alpha) if n[j] > 0 else 1.0
        if p <= eta:
            acc.append(j)
        else:
            break
    return max(acc) if acc else -1


def training_order(D_tr, s_tr, lams, delta, alpha, selective=True):
    """Order of grid indices by increasing binomial p-value on (cross-fitted) training data; ties favour aggressive."""
    o = first_stop(D_tr, s_tr, lams)
    k, n = risk_counts(o, delta, selective)
    p = np.array([binom.cdf(k[j], n[j], alpha) if n[j] > 0 else 1.0 for j in range(len(lams))])
    return np.lexsort((-np.arange(len(lams)), p))
