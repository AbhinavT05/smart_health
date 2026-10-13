# MobiFall reproduction run — stopped at the all-feature check

Run date: 2026-10-13. This report records completed checks and the first reproduced-number contradiction. Per the requested fail-stop rule, remaining experiment settings and system checks were not run.

## Setup and fixes made

- Removed the trainer's legacy CSV default and made its compatibility entry point call the native MobiFall experiment and bundle pipeline. A missing `data/raw/MobiFall` directory now raises `FileNotFoundError`.
- Replaced the old training CLI with an alias for `scripts/run_mobifall_pipeline.py`; removed the redundant legacy prediction script.
- Replaced generated-waveform simulation with real MobiFall replay. Replay messages carry `simulated: true`; manual fall injection and generated waveforms were removed.
- Added a pytest guard against non-test-source references to the removed CSV and generator; renamed the synthetic inference unit test accordingly.
- Updated README claims and frontend labels to describe a MobiFall v2.0 research prototype, not a medical device, and removed the unsupported accuracy claim.
- Added the `inspect`, `experiment`, `train`, and `all` pipeline commands. Each command writes a seed/config record under `reports/pipeline_configs/`.
- Removed obsolete UCI-specific logic from the subject splitter and changed the dataset availability check to require MobiFall only.

## Real-file validation output

Command: Python loader check on `data/raw/MobiFall/sub2/ADL/WAL/WAL_acc_2_1.txt` and its gyro pair.

```text
first_acc_g= (0.058594, 1.061523, -0.110352)
acc= 27224 rows 299.959 s median_dt_ms 10.537
gyro= 59996 rows 299.991 s median_dt_ms 4.997
aligned= 14997 samples last_t_s 299.92 NaNs 0
```

The first aligned-output command selected an incorrectly zero-padded internal trial ID and printed an empty selection. It was corrected to select by `subject_id == MOB_02` and the real trial suffix; the corrected output above matches the requested values.

Command: `python scripts/run_mobifall_pipeline.py inspect`

```text
Run config: reports/pipeline_configs/inspect.json
paired_trials=630; incomplete_trials=0
BSC: n=72 duration_s min=6.577 median=9.885 max=14.971
CSI: n=54 duration_s min=5.023 median=5.885 max=5.991
CSO: n=54 duration_s min=5.040 median=5.887 max=5.990
FKL: n=72 duration_s min=9.798 median=9.908 max=14.858
FOL: n=72 duration_s min=9.756 median=9.900 max=9.990
JOG: n=27 duration_s min=29.795 median=29.913 max=29.990
JUM: n=27 duration_s min=29.805 median=29.878 max=29.981
SCH: n=54 duration_s min=5.798 median=5.891 max=5.990
SDL: n=72 duration_s min=9.795 median=9.883 max=9.986
STD: n=9 duration_s min=299.811 median=299.871 max=299.993
STN: n=54 duration_s min=9.795 median=9.899 max=9.989
STU: n=54 duration_s min=7.151 median=9.893 max=9.986
WAL: n=9 duration_s min=299.803 median=299.901 max=299.990
global_median_trial_duration_s=9.875
global_median_trial_window_count=6
chosen_per_trial_window_cap=6
```

Command: `python scripts/build_mobifall_manifest.py`

```text
non_fall: subjects=9, trials=342, windows=1561
fall: subjects=24, trials=288, windows=469
fall subjects: MOB_01–MOB_21, MOB_29, MOB_30, MOB_31
ADL subjects: MOB_02, MOB_03, MOB_04, MOB_05, MOB_07, MOB_08, MOB_09, MOB_10, MOB_11
Cohort check: PASS
```

The first manifest check exposed an obsolete expected-subject list in the checker (`MOB_01`–`MOB_24`). The checker was corrected to the specified v2.0 cohort; observed real data matched the requested cohort and counts.

## First stopping condition: all-feature confusion counts

Command started from a deleted feature cache: `python -u scripts/run_mobifall_pipeline.py experiment`. Seed 42 was used with five subject-grouped folds. The all-feature set reported 142 columns. All five test folds included both classes.

Pooled confusion counts summed from the five newly written `reports/mobifall_folds/all_features_fold_*.json` files (TN/FP/FN/TP):

| Model | Expected | Reproduced | Difference |
|---|---:|---:|---:|
| SVM | 1553/8/12/457 | 1553/8/12/457 | Match |
| RF | 1542/19/18/451 | 1540/21/16/453 | Mismatch |
| Fused equal weights | 1552/9/13/456 | 1553/8/13/456 | Mismatch |

Printed pooled metrics included SVM accuracy `0.9901478` and F1 `0.9785867`, matching the requested rounded values. RF and fused confusion counts do not match the expected values, so the run was stopped before the orientation-invariant and paper-six-feature runs, label-shuffle/majority rerun, model-bundle retraining, tests, backend replay, website check, or Android build.

Likely cause (inference): `svmag_var` and `svmag_p2p` were added to feature extraction to support the requested six-feature paper set, expanding the existing all-feature model input from its prior 140 columns to 142. This changes RF/fusion decisions while leaving SVM's listed confusion counts unchanged. This is not confirmed; no attempt was made to exclude those columns and continue because the reproduced confusion counts already triggered the explicit stop condition.

## State and limitations at stop

- The grouped run wrote all-feature fold JSON files but was interrupted before writing fresh pooled `mobifall_metrics.json` / `results_table.csv`. Any pre-existing pooled report files in the workspace are from an earlier run and must not be treated as results for this partial run.
- `reports/system_check.md` was not produced. No unit suite, script tests, backend run, browser check, or Android build was started.
- `reports/dataset_manifest.json` contains the real manifest (630 trials, 2,030 windows, cap 6). `reports/pipeline_configs/inspect.json` and `experiment.json` record the dataset and fixed seed.
- The loader logs one repeated gyro timestamp in `sub10/ADL/STU/STU_gyro_10_2.txt`, which it collapses by averaging. No other mismatch in the required raw sample checks was observed.
