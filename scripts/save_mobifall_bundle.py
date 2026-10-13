"""Fit the backend-compatible hybrid bundle from MobiFall using fold-selected params."""

import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.training.mobifall_dataset import META_COLUMNS, _feature_hash


def main() -> None:
    cache = BASE_DIR / "data" / "processed" / f"mobifall_features_{_feature_hash()}.csv"
    metrics_path = BASE_DIR / "reports" / "mobifall_metrics.json"
    fold_dir = BASE_DIR / "reports" / "mobifall_folds"
    if not cache.exists() or not metrics_path.exists():
        raise FileNotFoundError("Run scripts/run_mobifall_experiments.py before saving the model bundle")
    features = __import__("pandas").read_csv(cache)
    cols = [c for c in features if c not in META_COLUMNS and __import__("pandas").api.types.is_numeric_dtype(features[c])]
    x = features[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    y = features.is_fall.to_numpy(dtype=int)

    fold_files = sorted(fold_dir.glob("all_features_fold_*.json"))
    if len(fold_files) != 5:
        raise ValueError(f"Expected 5 all-feature fold reports; found {len(fold_files)}")
    fold_reports = [json.loads(p.read_text(encoding="utf-8")) for p in fold_files]
    rf_params = dict(Counter(tuple(sorted(r["selected_rf_params"].items())) for r in fold_reports).most_common(1)[0][0])
    svm_params = dict(Counter(tuple(sorted(r["selected_svm_params"].items())) for r in fold_reports).most_common(1)[0][0])
    fusion = fold_reports[0]["validation_fusion"]
    weights_rf = float(np.median([r["validation_fusion"]["weight_rf"] for r in fold_reports]))
    threshold = float(np.median([r["validation_fusion"]["threshold"] for r in fold_reports]))
    weights_rf = round(weights_rf, 2)
    weights_svm = round(1.0 - weights_rf, 2)

    rf = RandomForestClassifier(**rf_params, random_state=42, n_jobs=1).fit(x, y)
    scaler = StandardScaler().fit(x)
    svm = CalibratedClassifierCV(
        estimator=SVC(C=svm_params["estimator__svm__C"], gamma=svm_params["estimator__svm__gamma"], random_state=42),
        method="sigmoid", cv=3, ensemble=False,
    ).fit(scaler.transform(x), y)
    classes = ["NON_FALL", "FALL"]
    bundle = {
        "bundle_format_version": 1,
        "architecture": "hybrid_rf_svm",
        "scikit_learn_version": sklearn.__version__,
        "rf_model": rf,
        "svm_model": svm,
        "scaler": scaler,
        "feature_names": cols,
        "classes": classes,
        "class_to_idx": {name: i for i, name in enumerate(classes)},
        "fall_class_idx": 1,
        "model_name": "MobiFall RF + calibrated SVM",
        "model_version": "mobifall-binary-1.0",
        "fusion": {
            "method": "probability_weighted_sum",
            "weight_rf": weights_rf,
            "weight_svm": weights_svm,
            "threshold": threshold,
            "selected_on": "outer-training-subject grouped OOF validation across folds",
        },
        "selected_rf_params": rf_params,
        "selected_svm_params": svm_params,
        "sampling_rate": 50.0,
        "window_size_seconds": 2.56,
        "window_samples": 128,
        "feature_setting": "all features",
        "trained_on_dataset": "MobiFall",
        "trained_subjects": sorted(features.subject_id.unique().tolist()),
        "random_seed": 42,
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8"))["all features"]["Fused validation-tuned"],
    }
    output = BASE_DIR / "data" / "models" / "fall_detection_model.joblib"
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    print(f"Saved backend-compatible model bundle: {output}")
    print(f"Input contract: window_samples=128 sampling_rate=50 Hz acc=g gyro=rad/s")
    print(f"RF params: {rf_params}")
    print(f"SVM params: {svm_params}")
    print(f"Fusion: weights=({weights_rf:.2f},{weights_svm:.2f}) threshold={threshold:.2f}")


if __name__ == "__main__":
    main()
