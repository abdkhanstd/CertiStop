"""Build per-test cleaned series and prefix feature tables for the Malaga treadmill GXT dataset."""
import numpy as np, pandas as pd, argparse, os

def clean_test(d):
    d = d.sort_values('time').reset_index(drop=True)
    sp = d.Speed.values
    vmax = sp.max()
    # end of exercise: last sample at max speed (recovery drops speed afterwards)
    i_end = np.where(sp >= vmax - 1e-9)[0].max()
    d = d.iloc[:i_end + 1].copy()
    # robust HR: rolling median over 5 breaths, drop implausible values
    hr = d.HR.where((d.HR > 40) & (d.HR < 230))
    d['HRs'] = hr.rolling(7, min_periods=1, center=False).median().ffill()  # causal: no back-filling
    for c in ['VO2', 'VCO2', 'VE', 'RR']:
        d[c + 's'] = d[c].rolling(7, min_periods=1).median()
    return d

def vo2max_of(d, w):
    # max 30-s time-averaged VO2 (mL/min) -> mL/kg/min
    s = d.set_index(pd.to_timedelta(d.time, unit='s')).VO2.rolling('30s').mean()
    return s.max() / w, s.idxmax().total_seconds()

def slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 5 or np.ptp(x) < 1e-6: return np.nan, np.nan
    b, a = np.polyfit(x, y, 1)
    return b, a

def prefix_features(d, t, info):
    p = d[d.time <= t]
    last = p[p.time > t - 30]
    age, w, h, sex = info['Age'], info['Weight'], info['Height'], info['Sex']
    hrmax_pred = 208 - 0.7 * age  # Tanaka
    f = dict(t=t, age=age, weight=w, height=h, sex=sex, bmi=w / (h / 100) ** 2)
    f['speed'] = p.Speed.iloc[-1]; f['speed_max'] = p.Speed.max()
    f['hr'] = last.HRs.mean(); f['hr_frac'] = f['hr'] / hrmax_pred
    f['hr_min'] = p.HRs.min()
    ex = p[p.Speed > 5.5]  # running/incremental part
    f['t_inc'] = (t - ex.time.iloc[0]) if len(ex) else 0.0
    b, a = slope(ex.Speed, ex.HRs) if len(ex) else (np.nan, np.nan)
    f['hr_speed_slope'] = b
    # gas-free extrapolation: speed at predicted HRmax, ACSM running VO2 at that speed
    if np.isfinite(b) and b > 0.5:
        v_at_hrmax = (hrmax_pred - a) / b
        f['acsm_extrap'] = 3.5 + 0.2 * v_at_hrmax * 1000 / 60
    else:
        f['acsm_extrap'] = np.nan
    f['acsm_now'] = 3.5 + 0.2 * f['speed'] * 1000 / 60
    r60 = p[p.time > t - 60]
    bh, _ = slope(r60.time, r60.HRs); f['hr_recent_slope'] = bh
    f['hrr_frac'] = (f['hr'] - f['hr_min']) / max(hrmax_pred - f['hr_min'], 1.0)
    if 'VO2s' in p and p.VO2.notna().sum() > 5:
        vo2 = p.VO2s / w
        f['vo2'] = last.VO2s.mean() / w; f['vo2_max_sofar'] = vo2.max()
        f['vco2'] = last.VCO2s.mean() / w; f['ve'] = last.VEs.mean(); f['rr'] = last.RRs.mean()
        f['rer'] = f['vco2'] / f['vo2'] if f['vo2'] > 0 else np.nan
        f['ve_vo2'] = f['ve'] / (f['vo2'] * w / 1000) if f['vo2'] > 0 else np.nan
        f['ve_vco2'] = f['ve'] / (f['vco2'] * w / 1000) if f['vco2'] > 0 else np.nan
        f['excess_co2'] = f['vco2'] - f['vo2']
        f['o2pulse'] = f['vo2'] * w / f['hr'] if f['hr'] > 0 else np.nan
        # ventilatory-equivalent markers of the first and second thresholds (nadir so far and rise above it)
        q = p[p.time > 60]
        if len(q) > 10:
            eq1 = (q.VEs / (q.VO2s / 1000)).rolling(15, min_periods=5).mean()
            eq2 = (q.VEs / (q.VCO2s / 1000)).rolling(15, min_periods=5).mean()
            f['ve_vo2_rise'] = f['ve_vo2'] - eq1.min(); f['ve_vco2_rise'] = f['ve_vco2'] - eq2.min()
            i1 = eq1.idxmin(); f['t_since_vt1'] = t - q.time.loc[i1] if np.isfinite(eq1.min()) else np.nan
            f['vo2_at_vt1'] = (q.VO2s.loc[i1] / w) if np.isfinite(eq1.min()) else np.nan
        b2, a2 = slope(ex.HRs, ex.VO2s / w) if len(ex) else (np.nan, np.nan)
        f['vo2_hr_slope'] = b2
        f['hr_extrap'] = a2 + b2 * hrmax_pred if np.isfinite(b2) else np.nan
        b3, a3 = slope(ex.Speed, ex.VO2s / w) if len(ex) else (np.nan, np.nan)
        f['vo2_speed_slope'] = b3
        b4, _ = slope(r60.time, r60.VO2s / w); f['vo2_recent_slope'] = b4
    return f

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--raw', default='data_raw/malaga'); ap.add_argument('--out', default='data/malaga_prefix.parquet')
    ap.add_argument('--step', type=float, default=30.0)
    a = ap.parse_args()
    m = pd.read_csv(os.path.join(a.raw, 'test_measure.csv')); s = pd.read_csv(os.path.join(a.raw, 'subject-info.csv')).set_index('ID_test')
    rows, tests = [], []
    for tid, d in m.groupby('ID_test'):
        info = s.loc[tid]
        if d.VO2.notna().sum() < 50: continue  # 30 tests have no gas data; target needs VO2
        d = clean_test(d)
        y, t_peak = vo2max_of(d, info['Weight'])
        T = d.time.iloc[-1]
        hrmax = d.HRs.max()
        tests.append(dict(ID_test=tid, ID=info['ID'], y=y, T=T, t_peak=t_peak, hrmax=hrmax, vmax=d.Speed.max(), n=len(d)))
        for t in np.arange(a.step * 2, T + 1e-9, a.step):
            f = prefix_features(d, t, info); f.update(ID_test=tid, ID=info['ID'], y=y, T=T, hrmax=hrmax)
            rows.append(f)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame(rows).to_parquet(a.out); pd.DataFrame(tests).to_csv(a.out.replace('_prefix.parquet', '_tests.csv'), index=False)
    print(len(tests), len(rows))
