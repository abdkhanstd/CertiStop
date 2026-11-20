"""Smoke test: one development split written to results/tmp (not part of the reported results)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import run

ds = sys.argv[1] if len(sys.argv) > 1 else 'dublin'
os.makedirs('results/tmp/parts', exist_ok=True); os.makedirs('weights/tmp', exist_ok=True)
P, F = run.load(ds)
t = time.time(); print(run.run_rep(ds, 98, P, F, 'results/tmp', 'weights/tmp'), round(time.time() - t))
S = pd.read_csv('results/tmp/parts/summ_rep98.csv')
pd.set_option('display.width', 200)
print(S[(S.delta == 5) & ((S.alpha == 0.1) | S.alpha.isna())][['input', 'rule', 'risk', 'early', 'saved_s']].round(3).to_string())
print(pd.read_csv('results/tmp/parts/timing_rep98.csv'))
