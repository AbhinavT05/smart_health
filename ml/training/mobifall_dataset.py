"""MobiFall-only 50 Hz windowing, labeling, feature caching, and manifests."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.loaders.mobifall import load_mobifall
from ml.features.feature_engineering import extract_features_from_window

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "MobiFall"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
WINDOW_SIZE = 128
STRIDE = 64
SAMPLE_RATE = 50.0
SENSOR_COLUMNS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
META_COLUMNS = ["window_id", "dataset", "subject_id", "trial_id", "activity_code", "is_fall", "window_start"]


def _feature_hash() -> str:
    source = inspect.getsource(extract_features_from_window)
    payload = f"{source}\nwindow={WINDOW_SIZE};stride={STRIDE};rate={SAMPLE_RATE};fall_center_tolerance_s=1"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _candidate_starts(length: int) -> list[int]:
    return list(range(0, length - WINDOW_SIZE + 1, STRIDE))


def _evenly_spaced(indices: list[int], cap: int) -> list[int]:
    if cap <= 0 or len(indices) <= cap:
        return indices
    positions = np.linspace(0, len(indices) - 1, cap, dtype=int)
    return [indices[i] for i in positions]


def build_mobifall_windows(
    root: str | Path = RAW_DIR,
    *,
    cache: bool = True,
    verbose: bool = False,
) -> tuple[pd.DataFrame, int]:
    """Build capped, binary-labeled windows; fall trials retain impact windows only."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"MobiFall dataset directory not found: {root}")
    cache_path = PROCESSED_DIR / f"mobifall_features_{_feature_hash()}.csv"
    cap_path = cache_path.with_suffix(".cap.json")
    if cache and cache_path.exists() and cap_path.exists():
        if verbose:
            print(f"Loading feature cache {cache_path}")
        return pd.read_csv(cache_path), int(json.loads(cap_path.read_text())["window_cap_per_trial"])

    if verbose:
        print(f"Loading and aligning real MobiFall trials from {root}")
    stream = load_mobifall(root)
    starts_by_trial: dict[str, list[int]] = {}
    trial_info = {}
    for trial_id, group in stream.groupby("trial_id", sort=True):
        group = group.sort_values("timestamp_s")
        starts = _candidate_starts(len(group))
        starts_by_trial[trial_id] = starts
        trial_info[trial_id] = (len(group), group.iloc[0])
    median_trial_windows = float(np.median([len(v) for v in starts_by_trial.values()]))
    window_cap = max(1, int(np.floor(median_trial_windows)))
    if verbose:
        print(f"Median trial window count={median_trial_windows:.1f}; chosen cap={window_cap}")

    records = []
    for trial_id, group in stream.groupby("trial_id", sort=True):
        group = group.sort_values("timestamp_s").reset_index(drop=True)
        values = group[SENSOR_COLUMNS].to_numpy(dtype=np.float64)
        starts = starts_by_trial[trial_id]
        meta = trial_info[trial_id][1]
        if int(meta.is_fall):
            svm = np.linalg.norm(values[:, :3], axis=1)
            impact = int(np.argmax(svm))
            # Window centers within +/- one second of the maximum SVMag impact.
            positive_starts = [
                start for start in starts
                if abs((start + (WINDOW_SIZE - 1) / 2) - impact) <= SAMPLE_RATE
            ]
            selected = _evenly_spaced(positive_starts, window_cap)
        else:
            selected = _evenly_spaced(starts, window_cap)
        for start in selected:
            window = values[start : start + WINDOW_SIZE]
            features = extract_features_from_window(*[window[:, i] for i in range(6)], sampling_rate=SAMPLE_RATE)
            features.update({
                "window_id": f"{trial_id}_w{start}",
                "dataset": "MobiFall",
                "subject_id": meta.subject_id,
                "trial_id": trial_id,
                "activity_code": meta.activity_code,
                "is_fall": int(meta.is_fall),
                "window_start": int(start),
            })
            records.append(features)
    features_df = pd.DataFrame(records)
    if features_df.empty:
        raise ValueError("No windows were created from the supplied MobiFall trials")
    if cache:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(cache_path, index=False)
        cap_path.write_text(json.dumps({"window_cap_per_trial": window_cap}, indent=2), encoding="utf-8")
        if verbose:
            print(f"Saved feature cache {cache_path}")
    return features_df, window_cap


def manifest_for_windows(features: pd.DataFrame) -> dict:
    """Summarize observed subjects, trials, and windows by binary class."""
    classes = {}
    for label, name in ((0, "non_fall"), (1, "fall")):
        part = features[features.is_fall == label]
        classes[name] = {
            "subjects": int(part.subject_id.nunique()),
            "subject_ids": sorted(part.subject_id.unique().tolist()),
            "trials": int(part.trial_id.nunique()),
            "windows": int(len(part)),
        }
    by_activity = {}
    for code, part in features.groupby("activity_code"):
        by_activity[str(code)] = {
            "class": "fall" if int(part.is_fall.iloc[0]) else "non_fall",
            "subjects": int(part.subject_id.nunique()),
            "trials": int(part.trial_id.nunique()),
            "windows": int(len(part)),
        }
    adl_subjects = set(features.loc[features.is_fall == 0, "subject_id"])
    fall_subjects = set(features.loc[features.is_fall == 1, "subject_id"])
    shared = adl_subjects & fall_subjects
    fall_only = fall_subjects - adl_subjects
    adl_trials = features[features.is_fall == 0].drop_duplicates("trial_id")
    fall_trials = features[features.is_fall == 1].drop_duplicates("trial_id")
    fall_trials_by_cohort = {
        "subjects_also_with_adls": int(fall_trials[fall_trials.subject_id.isin(shared)].trial_id.nunique()),
        "fall_only_subjects": int(fall_trials[fall_trials.subject_id.isin(fall_only)].trial_id.nunique()),
    }
    return {
        "dataset": "MobiFall",
        "sample_rate_hz": SAMPLE_RATE,
        "window_samples": WINDOW_SIZE,
        "window_seconds": WINDOW_SIZE / SAMPLE_RATE,
        "stride_samples": STRIDE,
        "window_cap_per_trial": int(features.attrs.get("window_cap_per_trial", -1)),
        "classes": classes,
        "activity_codes": by_activity,
        "cohort_relationship": {
            "shared_subjects": sorted(shared),
            "fall_only_subjects": sorted(fall_only),
            "adl_only_subjects": sorted(adl_subjects - fall_subjects),
            "shared_subject_count": len(shared),
            "fall_only_subject_count": len(fall_only),
            "fall_trials_by_cohort": fall_trials_by_cohort,
            "interpretation": (
                "ADL trials are all from subjects that also have fall trials. "
                "Most fall trials are from the fall-only subject cohort; report this cohort composition "
                "because it affects cross-subject generalization."
            ),
        },
    }


def write_manifest(features: pd.DataFrame, window_cap: int) -> Path:
    features.attrs["window_cap_per_trial"] = window_cap
    manifest = manifest_for_windows(features)
    manifest["window_cap_per_trial"] = window_cap
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "dataset_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
