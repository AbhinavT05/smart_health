"""
FallGuard AI - Hybrid Random Forest + SVM training pipeline.

Trains exactly two classifiers (Random Forest and SVM) on the same features,
selects fusion weights and threshold on a validation split only, and serializes
both models with the fitted preprocessing pipeline.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from ml.evaluation.evaluator import (
    FIGURES_DIR,
    REPORTS_DIR,
    evaluate_classifier,
    plot_activity_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
)
from ml.features.feature_engineering import extract_dataset_features
from ml.inference.hybrid_fusion import select_fusion_parameters
from ml.preprocessing.cleaner import clean_sensor_stream
from ml.preprocessing.splitter import split_by_subject, split_train_val_from_train

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ModelTrainer")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MODEL_PATH = MODELS_DIR / "fall_detection_model.joblib"
METRICS_JSON_PATH = REPORTS_DIR / "model_metrics.json"
COMPARISON_JSON_PATH = REPORTS_DIR / "model_comparison_report.json"
FEATURE_CACHE_PATH = PROCESSED_DIR / "extracted_features_hybrid_v1.csv"

RANDOM_SEED = 42
META_COLS = ["window_id", "subject_id", "activity", "is_fall", "split_origin"]


def _matrices(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.nan_to_num(df[feature_cols].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y_activity = df["activity"].to_numpy()
    y_fall = df["is_fall"].to_numpy().astype(int)
    return x, y_activity, y_fall


def _fall_proba(model, x: np.ndarray, class_names: List[str], fall_name: str = "FALL") -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise RuntimeError(f"{type(model).__name__} does not provide predict_proba; cannot fuse.")
    probs = model.predict_proba(x)
    if fall_name not in class_names:
        return np.zeros(len(x), dtype=float)
    idx = class_names.index(fall_name)
    return probs[:, idx]


def train_and_evaluate_all_models(
    dataset_csv_path: Path = None,
    sampling_rate: float = 50.0,
) -> Dict[str, Any]:
    """Train Random Forest and SVM, fuse on validation, evaluate on held-out test subjects."""
    logger.info("=" * 70)
    logger.info("  FALLGUARD AI - HYBRID RANDOM FOREST + SVM TRAINING PIPELINE")
    logger.info("=" * 70)

    if dataset_csv_path is None:
        dataset_csv_path = PROCESSED_DIR / "unified_sensor_stream.csv"

    if not dataset_csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_csv_path}. "
            "Run `python data/download_datasets.py` first. Training was not performed."
        )

    logger.info(f"Loading raw sensor stream from {dataset_csv_path}...")
    df_raw = pd.read_csv(dataset_csv_path)
    required = {"acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "activity", "is_fall", "subject_id", "window_id"}
    missing = required - set(df_raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if len(df_raw) < 200:
        raise ValueError(
            f"Dataset is insufficient ({len(df_raw)} rows). Need a real labeled IMU stream before training."
        )

    df_clean = clean_sensor_stream(df_raw)

    if FEATURE_CACHE_PATH.exists():
        logger.info(f"Loading cached window features from {FEATURE_CACHE_PATH}...")
        df_features = pd.read_csv(FEATURE_CACHE_PATH)
    else:
        logger.info("Extracting statistical, SVMag, SMA, and frequency features from sliding windows...")
        df_features = extract_dataset_features(df_clean, sampling_rate=sampling_rate)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(FEATURE_CACHE_PATH, index=False)
        logger.info(f"Saved extracted features cache to {FEATURE_CACHE_PATH}")

    if "FALL" not in set(df_features["activity"].unique()) and int(df_features["is_fall"].sum()) == 0:
        raise ValueError("Dataset contains no FALL labels. Hybrid fall detection cannot be trained.")

    plot_activity_distribution(df_features, FIGURES_DIR / "activity_distribution.png")

    logger.info("Partitioning data with subject-independent train/test splitting...")
    train_all_df, test_df, train_subjects, test_subjects = split_by_subject(
        df_features, test_subject_ratio=0.30, random_seed=RANDOM_SEED
    )
    fit_df, val_df, fit_subjects, val_subjects = split_train_val_from_train(
        train_all_df, val_subject_ratio=0.20, random_seed=RANDOM_SEED
    )

    feature_cols = [c for c in fit_df.columns if c not in META_COLS]
    if not feature_cols:
        raise ValueError("No numeric feature columns were produced.")

    x_fit, y_act_fit_raw, y_fall_fit = _matrices(fit_df, feature_cols)
    x_val, y_act_val_raw, y_fall_val = _matrices(val_df, feature_cols)
    x_test, y_act_test_raw, y_fall_test = _matrices(test_df, feature_cols)

    label_encoder = LabelEncoder()
    y_fit = label_encoder.fit_transform(y_act_fit_raw)
    y_val = label_encoder.transform(y_act_val_raw)
    y_test = label_encoder.transform(y_act_test_raw)
    class_names = list(label_encoder.classes_)
    if "FALL" not in class_names:
        raise ValueError("FALL class missing after encoding. Check activity labels.")

    logger.info(f"Classes ({len(class_names)}): {class_names}")
    logger.info(f"Fit samples: {len(x_fit)} subjects={fit_subjects}")
    logger.info(f"Val samples: {len(x_val)} subjects={val_subjects}")
    logger.info(f"Test samples: {len(x_test)} subjects={test_subjects}")

    scaler = StandardScaler()
    x_fit_scaled = scaler.fit_transform(x_fit)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=16,
        min_samples_split=4,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    svm_base = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    # Platt scaling fitted with internal CV on the training partition only.
    svm = CalibratedClassifierCV(svm_base, method="sigmoid", cv=3)

    logger.info("Training Random Forest on unscaled features...")
    rf.fit(x_fit, y_fit)
    logger.info("Training calibrated SVM on scaled features...")
    svm.fit(x_fit_scaled, y_fit)

    p_rf_val = _fall_proba(rf, x_val, class_names)
    p_svm_val = _fall_proba(svm, x_val_scaled, class_names)

    rf_brier = float(brier_score_loss(y_fall_val, p_rf_val))
    svm_brier = float(brier_score_loss(y_fall_val, p_svm_val))
    logger.info(f"Validation Brier score — RF: {rf_brier:.4f}, SVM: {svm_brier:.4f}")
    if rf_brier > 0.35 or svm_brier > 0.35:
        logger.warning(
            "One or both probability estimates have a high Brier score on validation. "
            "Fusion still uses probability-level combination; inspect reports/model_metrics.json."
        )

    fusion_params = select_fusion_parameters(p_rf_val, p_svm_val, y_fall_val)
    logger.info(
        "Selected fusion (validation only): "
        f"w_RF={fusion_params['weight_rf']}, w_SVM={fusion_params['weight_svm']}, "
        f"threshold={fusion_params['threshold']}"
    )

    rf_eval = evaluate_classifier(rf, x_test, y_test, class_names, fall_class_name="FALL")
    svm_eval = evaluate_classifier(svm, x_test_scaled, y_test, class_names, fall_class_name="FALL")

    p_rf_test = _fall_proba(rf, x_test, class_names)
    p_svm_test = _fall_proba(svm, x_test_scaled, class_names)
    p_hybrid_test = (
        fusion_params["weight_rf"] * p_rf_test + fusion_params["weight_svm"] * p_svm_test
    )
    y_hybrid_fall = (p_hybrid_test >= fusion_params["threshold"]).astype(int)

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    hybrid_acc = float(accuracy_score(y_fall_test, y_hybrid_fall))
    hybrid_prec = float(precision_score(y_fall_test, y_hybrid_fall, zero_division=0))
    hybrid_rec = float(recall_score(y_fall_test, y_hybrid_fall, zero_division=0))
    hybrid_f1 = float(f1_score(y_fall_test, y_hybrid_fall, zero_division=0))
    hybrid_auc = (
        float(roc_auc_score(y_fall_test, p_hybrid_test)) if len(np.unique(y_fall_test)) > 1 else None
    )
    hybrid_cm = confusion_matrix(y_fall_test, y_hybrid_fall, labels=[0, 1]).tolist()

    print("\n" + "=" * 80)
    print(f"{'Model':<24} | {'Accuracy':<9} | {'Fall Prec':<10} | {'Fall Rec':<10} | {'Fall F1':<9} | {'AUC'}")
    print("-" * 80)
    for name, ev in (("Random Forest", rf_eval), ("SVM (calibrated)", svm_eval)):
        auc = ev["fall_roc_auc"]
        auc_s = f"{auc*100:5.2f}%" if auc is not None else "  n/a"
        print(
            f"{name:<24} | {ev['accuracy']*100:6.2f}%   | {ev['fall_precision']*100:6.2f}%   | "
            f"{ev['fall_recall']*100:6.2f}%    | {ev['fall_f1']*100:6.2f}%  | {auc_s}"
        )
    auc_s = f"{hybrid_auc*100:5.2f}%" if hybrid_auc is not None else "  n/a"
    print(
        f"{'Hybrid RF+SVM':<24} | {hybrid_acc*100:6.2f}%   | {hybrid_prec*100:6.2f}%   | "
        f"{hybrid_rec*100:6.2f}%    | {hybrid_f1*100:6.2f}%  | {auc_s}"
    )
    print("=" * 80)

    comparison_rows = [
        {
            "Model": "Random Forest",
            "Accuracy": rf_eval["accuracy"],
            "Macro F1": rf_eval["macro_f1"],
            "Fall Recall": rf_eval["fall_recall"],
            "Fall Precision": rf_eval["fall_precision"],
            "Weighted F1": rf_eval["weighted_f1"],
            "Fall FPR": rf_eval["fall_fpr"],
            "Fall FNR": rf_eval["fall_fnr"],
            "Fall ROC-AUC": rf_eval["fall_roc_auc"],
        },
        {
            "Model": "Support Vector Machine",
            "Accuracy": svm_eval["accuracy"],
            "Macro F1": svm_eval["macro_f1"],
            "Fall Recall": svm_eval["fall_recall"],
            "Fall Precision": svm_eval["fall_precision"],
            "Weighted F1": svm_eval["weighted_f1"],
            "Fall FPR": svm_eval["fall_fpr"],
            "Fall FNR": svm_eval["fall_fnr"],
            "Fall ROC-AUC": svm_eval["fall_roc_auc"],
        },
        {
            "Model": "Hybrid RF+SVM",
            "Accuracy": hybrid_acc,
            "Macro F1": hybrid_f1,
            "Fall Recall": hybrid_rec,
            "Fall Precision": hybrid_prec,
            "Weighted F1": hybrid_f1,
            "Fall FPR": None,
            "Fall FNR": 1.0 - hybrid_rec if hybrid_rec is not None else None,
            "Fall ROC-AUC": hybrid_auc,
        },
    ]
    df_comparison = pd.DataFrame(comparison_rows)
    plot_model_comparison(df_comparison, FIGURES_DIR / "model_comparison_bar.png")
    plot_confusion_matrix(hybrid_cm, ["NON_FALL", "FALL"], "Hybrid RF+SVM", FIGURES_DIR / "confusion_matrix_best.png")

    feat_importance = {}
    if hasattr(rf, "feature_importances_"):
        feat_importance = {feat: float(imp) for feat, imp in zip(feature_cols, rf.feature_importances_)}
        plot_feature_importance(feat_importance, top_n=20, save_path=FIGURES_DIR / "feature_importance_top20.png")

    def _sanitize_eval(ev: Dict[str, Any]) -> Dict[str, Any]:
        skip = {"y_pred", "y_prob", "classification_report"}
        return {k: v for k, v in ev.items() if k not in skip}

    hybrid_metrics = {
        "accuracy": hybrid_acc,
        "fall_precision": hybrid_prec,
        "fall_recall": hybrid_rec,
        "fall_f1": hybrid_f1,
        "fall_roc_auc": hybrid_auc,
        "confusion_matrix": hybrid_cm,
        "macro_precision": hybrid_prec,
        "macro_recall": hybrid_rec,
        "macro_f1": hybrid_f1,
        "weighted_f1": hybrid_f1,
        "fall_fnr": float(1.0 - hybrid_rec),
        "fall_fpr": None,
    }

    model_bundle = {
        "architecture": "hybrid_rf_svm",
        "rf_model": rf,
        "svm_model": svm,
        "scaler": scaler,
        "feature_names": feature_cols,
        "classes": class_names,
        "class_to_idx": {name: i for i, name in enumerate(class_names)},
        "fall_class_idx": class_names.index("FALL"),
        "model_name": "Hybrid Random Forest + SVM",
        "model_version": "2.0.0-hybrid-rf-svm",
        "fusion": {
            "method": "probability_weighted_sum",
            "formula": "P_hybrid = w_RF * P_RF(Fall) + w_SVM * P_SVM(Fall)",
            "weight_rf": fusion_params["weight_rf"],
            "weight_svm": fusion_params["weight_svm"],
            "threshold": fusion_params["threshold"],
            "min_weight": 0.20,
            "selected_on": "validation_split_only",
            "validation_metrics": {
                "fall_f1": fusion_params["validation_fall_f1"],
                "fall_recall": fusion_params["validation_fall_recall"],
                "fall_precision": fusion_params["validation_fall_precision"],
                "fall_roc_auc": fusion_params["validation_fall_roc_auc"],
            },
            "probability_reliability": {
                "rf_brier_validation": rf_brier,
                "svm_brier_validation": svm_brier,
                "svm_calibration": "CalibratedClassifierCV sigmoid, cv=3, training partition only",
            },
        },
        "metrics": hybrid_metrics,
        "rf_metrics": _sanitize_eval(rf_eval),
        "svm_metrics": _sanitize_eval(svm_eval),
        "feature_importance": feat_importance,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sampling_rate": sampling_rate,
        "window_size_seconds": 2.56,
        "window_samples": int(sampling_rate * 2.56),
        "random_seed": RANDOM_SEED,
        "train_subjects": list(fit_subjects),
        "val_subjects": list(val_subjects),
        "test_subjects": list(test_subjects),
        "total_train_windows": len(x_fit),
        "total_val_windows": len(x_val),
        "total_test_windows": len(x_test),
    }

    joblib.dump(model_bundle, FINAL_MODEL_PATH)
    logger.info(f"Saved hybrid model bundle to {FINAL_MODEL_PATH}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(hybrid_metrics, f, indent=2)
    with open(COMPARISON_JSON_PATH, "w") as f:
        json.dump(comparison_rows, f, indent=2)

    logger.info(f"Saved evaluation metrics to {METRICS_JSON_PATH}")
    return model_bundle
