# CertiStop: risk-controlled early termination of maximal exercise tests

Code and results for the paper *CertiStop: Risk-Controlled Early Termination of Maximal Exercise Tests for Evidence-Based Fitness Evaluation*.

CertiStop decides during a graded exercise test (GXT) when the anytime estimate of maximal oxygen uptake (VO2max) may be reported and the test stopped. It has four parts:

1. **Anytime predictor.** A gradient-boosting predictor that, with laboratory inputs, learns the gap between VO2max and the running peak oxygen uptake.
2. **Gate.** Stopping is eligible only once heart rate reaches 77 % of the age-predicted maximal heart rate.
3. **Stop score.** Selected on training data among candidates: the predictor's 80 % interval width, timing since the gate, elapsed time, %HRmax, RER or blood lactate, a VO2 plateau score and estimate stability. Each candidate is certified on half of the training participants after being ordered on the other half, over five halvings.
4. **Certified threshold.** A fixed-sequence Learn-then-Test procedure with binomial p-values, ordered by training p-values. It guarantees, with probability at least 1 - eta, that the error rate among stopped tests is at most alpha (per-stop guarantee).

A budget guarantee on the share of all tests stopped with a large error is also implemented, for analysing single scores.

## Data

Both datasets are public and are downloaded, not redistributed:

* Malaga treadmill maximal exercise tests, PhysioNet `treadmill-exercise-cardioresp` 1.0.1 (doi:10.13026/7ezk-j442)
* Dublin graded incremental test data, Zenodo record 10841412 (doi:10.5281/zenodo.10841412, CC BY 4.0)

## Reproduce

```bash
pip install -r requirements.txt
python certistop/download_data.py
python certistop/prep_malaga.py                 # data/malaga_prefix.parquet (30-s decision points)
python certistop/prep_dublin.py                 # data/dublin_info.csv, data/dublin_stages.csv
python certistop/feats_dublin.py                # data/dublin_prefix.parquet (per-stage decisions, pseudo-participants from sex + date of birth)
python certistop/run.py --ds malaga --reps 20 --jobs 10
python certistop/run.py --ds dublin --reps 20 --jobs 8
CERTISTOP_GATE=0.70 CERTISTOP_LIGHT=1 python certistop/run.py --ds malaga   # gate sensitivity (also 0.85, and for dublin), LIGHT keeps the full candidate set but only laboratory inputs and delta = 5
python certistop/seqnn.py --ds malaga --arch gru --gpu 0                    # GRU / Transformer baselines (also --arch tf, --ds dublin)
python certistop/extra.py                       # Dublin leave-one-sport-out
python certistop/export_curves.py --ds malaga   # risk-coverage curves (also dublin)
python certistop/analyze.py                     # results/summary/*.csv
python certistop/analyze_r3.py                  # pooled errors, permutation null, subgroup audit, Holm family of the stop-rule table
python certistop/analyze_r4.py                  # per-split permutation tests, interpolated risk-coverage curves
python certistop/analyze_r5.py                  # permutation null of fixed scores, uncertified rules, comparison at the 85 % end point's time saved
```

Seeds 0-19 are the reported participant-grouped splits (50 % train, 25 % calibration, 25 % test). Seeds 90-99 were used only for the early design studies in `dev_predictor.py`, `dev_stopscore.py` and `dev_stopscore2.py`.

## Files

| file | content |
|---|---|
| `certistop/core.py` | replay of a stopping rule on complete tests, fixed-sequence Learn-then-Test (natural order and training-ordered) |
| `certistop/models.py` | anytime predictor (residual or direct GBM, quantile GBMs, ridge) |
| `certistop/run.py` | full experiment: predictors, stop scores, budget and per-stop certification, score selection, baselines, ablations, per-sex and one-test-per-participant certification |
| `certistop/seqnn.py` | causal GRU and Transformer baselines |
| `certistop/extra.py` | Dublin leave-one-sport-out |
| `certistop/analyze.py`, `analyze_r3.py` | aggregation, corrected resampled t-tests, Holm, participant bootstrap, permutation null |
| `certistop/analyze_r4.py`, `analyze_r5.py` | per-split permutation tests (CertiStop and fixed scores, Holm over the CertiStop tests), risk-coverage curves on a fine grid, comparison with the 85 % HRmax end point at matched time saved |
| `results/<ds>/stop_summary.csv` | one row per split, input set, delta, alpha and stop rule (time saved, risk, number stopped, number with error above delta) |
| `results/<ds>/stop_tests.parquet` | per-test outcome (stopped, error, time saved, HR at stop) |
| `results/<ds>/pred_rows.parquet` | per-decision predictions of all predictors on the test splits |
| `results/<ds>_gate70`, `<ds>_gate85` | gate-sensitivity runs (per-split summaries only, models in the release) |
| `results/summary/` | aggregated tables, including `r3_sel_tests.csv` (significance marks of the stop-rule table) and the `r4_*` and `r5_*` analyses |

Trained models of all 20 splits (main runs and gate runs) are attached to the GitHub release as `weights.zip`. The reported numbers are in `results/summary/*.csv`, `results/<ds>/stop_summary.csv` and `timing.csv`, the gate runs in `results/<ds>_gate*/stop_summary.csv` and the leave-one-sport-out analysis in `results/extra/dublin_loso.csv`. The scripts that draw the paper's figures and typeset its tables belong to the manuscript and are not part of this repository.

## Licence

MIT for the code. The datasets keep their own licences.
