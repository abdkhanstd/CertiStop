"""Prefix (after-each-stage) feature table for the Dublin graded incremental test dataset.

Input: data/dublin_info.csv and data/dublin_stages.csv written by prep_dublin.py.
Output: data/dublin_prefix.parquet (one row per test and completed exercise stage) and data/dublin_tests.csv.
A decision is taken after each completed exercise stage k = 1..T-1, where T is the number of exercise stages
(the last stage is the exhaustion stage). Stages last 3 min (cycling, running) or 3 min plus 1 min rest
(rowing, kayak, discontinuous protocol).
"""
import numpy as np, pandas as pd, argparse

SPORTS = {'Cycling': 0, 'Running': 1, 'Rowing': 2, 'Kayak': 3}
STAGE_SEC = {'Cycling': 180, 'Running': 180, 'Rowing': 240, 'Kayak': 240}


def slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 2 or np.ptp(x) == 0:
        return np.nan, np.nan
    return np.polyfit(x, y, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--info', default='data/dublin_info.csv')
    ap.add_argument('--stages', default='data/dublin_stages.csv')
    ap.add_argument('--out', default='data/dublin_prefix.parquet')
    a = ap.parse_args()
    I = pd.read_csv(a.info); S = pd.read_csv(a.stages)
    # cleaning: plausible target range, known age, recognised sport
    I = I[(I.vo2max > 25) & (I.vo2max < 95) & I.age.notna() & I.sport.isin(SPORTS)].copy()
    S.loc[(S.hr < 35) | (S.hr > 230), 'hr'] = np.nan
    # fixed plausibility bounds only (the target is not used in preprocessing)
    S.loc[(S.vo2 <= 2) | (S.vo2 > 95), 'vo2'] = np.nan
    S.loc[(S.k == 0) & (S.vo2 > 15), 'vo2'] = np.nan
    S.loc[(S.bla <= 0) | (S.bla > 30), 'bla'] = np.nan
    rows, tests = [], []
    for r in I.itertuples():
        st = S[S.file == r.file].sort_values('k')
        st = st[st.load.notna()]
        n = len(st); T = n - 1  # exercise stages
        if T < 3:
            continue
        hm = 208 - 0.7 * r.age
        rest = st.iloc[0]; ex = st[st.k >= 1]
        sec = STAGE_SEC[r.sport]
        hr_meas_max = np.nanmax([r.hrmax, ex.hr.max()]) if np.isfinite(r.hrmax) else ex.hr.max()
        # pseudo-participant identifier: sex and date of birth (files carry no participant ID)
        pid = f'{r.sex}_{r.dob}' if isinstance(r.dob, str) and len(r.dob) == 10 else r.file
        tests.append(dict(ID_test=r.file, ID=pid, y=r.vo2max, T=T, Tsec=T * sec, sport=r.sport, sex=r.sex, age=r.age, hrmax=hr_meas_max))
        for k in range(1, T):
            p = ex[ex.k <= k]; l = p.iloc[-1]
            prev = p.iloc[-2] if len(p) > 1 else rest
            f = dict(ID_test=r.file, ID=pid, step=k, tsec=k * sec, Tsec=T * sec, y=r.vo2max, hrmax=hr_meas_max,
                     age=r.age, sexm=int(r.sex == 'M'), sport=SPORTS[r.sport], height=r.height, weight=r.weight, bmi=r.bmi,
                     stage=k, load=l.load, dload=l.load - prev.load,
                     hr=l.hr, dhr=l.hr - prev.hr, hr_rest=rest.hr, hr_frac=l.hr / hm,
                     hrr_frac=(l.hr - rest.hr) / max(hm - rest.hr, 1.0) if np.isfinite(rest.hr) else np.nan,
                     vo2=l.vo2, dvo2=l.vo2 - prev.vo2, vo2_rest=rest.vo2, vo2_max_sofar=p.vo2.max(),
                     bla=l.bla, dbla=l.bla - prev.bla, bla_rest=rest.bla)
            b, a0 = slope(p.hr.values, p.vo2.values)
            f['vo2_hr_slope'] = b; f['hr_extrap'] = a0 + b * hm if np.isfinite(b) else np.nan
            f['hr_load_slope'] = slope(p.load.values, p.hr.values)[0]
            f['vo2_load_slope'] = slope(p.load.values, p.vo2.values)[0]
            f['lbla_load_slope'] = slope(p.load.values, np.log(np.clip(p.bla.values, 0.3, None)))[0]
            f['o2pulse'] = l.vo2 * r.weight / l.hr if l.hr and np.isfinite(l.hr) else np.nan
            rows.append(f)
    P = pd.DataFrame(rows); Tt = pd.DataFrame(tests)
    P.to_parquet(a.out); Tt.to_csv(a.out.replace('_prefix.parquet', '_tests.csv'), index=False)
    print(P.shape, Tt.shape, Tt.sport.value_counts().to_dict())


if __name__ == '__main__':
    main()
