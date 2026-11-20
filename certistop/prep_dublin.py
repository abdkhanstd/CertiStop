"""Parse the Dublin graded incremental test files (Zenodo 10.5281/zenodo.10841412) into tidy tables."""
import pandas as pd, numpy as np, glob, os, argparse, warnings
warnings.filterwarnings('ignore')
ap = argparse.ArgumentParser(); ap.add_argument('--raw', default='data_raw/dublin/ex/data'); ap.add_argument('--out', default='data/dublin')
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
info_rows, stage_rows = [], []
for f in sorted(glob.glob(os.path.join(a.raw, 'A*.xlsx'))):
    fid = os.path.basename(f)[:-5]
    x = pd.read_excel(f, sheet_name=None, header=None)
    sh = list(x.values())
    if len(sh) < 3: continue
    kv = lambda d: dict(zip(d[0].astype(str).str.strip().str.rstrip(':'), d[1])) if d.shape[1] >= 2 else {}
    i1 = kv(sh[0]); i3 = kv(sh[2])
    if sh[1].shape[0] < 2 or sh[1].shape[1] < 3: print('skip', fid, [s_.shape for s_ in sh]); continue
    st = sh[1].copy(); st.columns = [str(c).strip() for c in st.iloc[0]]; st = st.iloc[1:].apply(pd.to_numeric, errors='coerce')
    st = st.dropna(how='all')
    load_col = st.columns[0]
    try:
        age = (pd.to_datetime(i1.get('Test Date')) - pd.to_datetime(i1.get('DOB'))).days / 365.25
    except Exception:
        age = np.nan
    try:
        dob = pd.to_datetime(i1.get('DOB')).strftime('%Y-%m-%d')
    except Exception:
        dob = ''
    info_rows.append(dict(file=fid, dob=dob, sex=str(i1.get('Gender', '')).strip(), sport=str(i1.get('Test', '')).strip(), event=str(i1.get('Event','')).strip(), age=age,
                          height=i1.get('Height'), weight=i1.get('Weight'), bmi=i1.get('BMI'), skinf=i1.get('Sum of skinfolds'), fat=i1.get('Fat'),
                          lbm=i1.get('LBM'), hb=i1.get('Hb'), hct=i1.get('Hct'), rbc=i1.get('RBC'), wbc=i1.get('WBC'),
                          vo2max=i3.get('VO2max'), vo2max_abs=i3.get('VO2max (absolute)'), hrmax=i3.get('Hrmax'), load_name=load_col, n_stage=len(st)))
    for k, r in enumerate(st.itertuples(index=False)):
        stage_rows.append(dict(file=fid, k=k, load=r[0], hr=r[1], vo2=r[2], bla=r[3], cad=r[4] if len(r) > 4 else np.nan))
I = pd.DataFrame(info_rows); S = pd.DataFrame(stage_rows)
for c in ['age','height','weight','bmi','skinf','fat','lbm','hb','hct','rbc','wbc','vo2max','vo2max_abs','hrmax']: I[c] = pd.to_numeric(I[c], errors='coerce')
I.to_csv(a.out + '_info.csv', index=False); S.to_csv(a.out + '_stages.csv', index=False)
print(I.shape, S.shape); print(I.sport.value_counts().to_dict(), I.sex.value_counts().to_dict(), I.load_name.value_counts().to_dict())
print(I.describe().round(2).to_string()); print(I.n_stage.value_counts().sort_index().to_dict())
