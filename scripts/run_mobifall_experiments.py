"""Subject-wise MobiFall-only RF/SVM/fusion evaluation with pooled OOF reports."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from scipy.stats import binomtest
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.inference.hybrid_fusion import fuse_fall_probability, select_fusion_parameters
from ml.training.mobifall_dataset import META_COLUMNS, build_mobifall_windows, write_manifest

REPORTS_DIR = BASE_DIR / "reports"
FOLDS_DIR = REPORTS_DIR / "mobifall_folds"
FIGURES_DIR = REPORTS_DIR / "figures"
N_OUTER_FOLDS = 5
SEED = 42


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, y_pred_override=None) -> dict:
    """Compute fall-positive binary metrics and count confusion cells."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = np.asarray(y_pred_override, dtype=int) if y_pred_override is not None else (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else None,
        "fnr": float(fn / (fn + tp)) if fn + tp else None,
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "class_counts": {"non_fall": int((y_true == 0).sum()), "fall": int((y_true == 1).sum())},
    }


def _models_and_grids(seed: int):
    rf = RandomForestClassifier(random_state=seed, n_jobs=1)
    rf_grid = {
        "n_estimators": [100, 250],
        "max_depth": [None, 16],
        "min_samples_leaf": [1, 3],
        "class_weight": ["balanced", "balanced_subsample"],
    }
    calibrated_svc = Pipeline([
        ("scaler", StandardScaler()),
        # Omit the deprecated `probability` argument; sigmoid calibration below
        # supplies Platt probabilities while SVC uses its decision function.
        ("svm", SVC(random_state=seed)),
    ])
    svm = CalibratedClassifierCV(estimator=calibrated_svc, method="sigmoid", cv=3, ensemble=False)
    svm_grid = {"estimator__svm__C": [0.1, 1, 10], "estimator__svm__gamma": ["scale", "auto"]}
    return rf, rf_grid, svm, svm_grid


def _checked_group_splits(x, y, groups, n_splits: int):
    splitter = GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(x, y, groups=groups))
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        train_subjects = set(groups[train_idx])
        test_subjects = set(groups[test_idx])
        if train_subjects & test_subjects:
            raise AssertionError(f"Subject leakage in fold {fold}: {sorted(train_subjects & test_subjects)}")
        if set(np.unique(y[test_idx])) != {0, 1}:
            raise ValueError(f"Fold {fold} test set lacks falls or ADLs")
    return splits


def _metrics_for_settings(features: pd.DataFrame, setting: str, feature_cols: list[str]) -> tuple[dict, list[dict]]:
    xdf = features[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = xdf.to_numpy(dtype=np.float64)
    y = features.is_fall.to_numpy(dtype=int)
    groups = features.subject_id.to_numpy()
    trials = features.trial_id.to_numpy()
    outer_splits = _checked_group_splits(x, y, groups, N_OUTER_FOLDS)
    oof = {"RF": np.full(len(y), np.nan), "SVM": np.full(len(y), np.nan), "Fused equal": np.full(len(y), np.nan), "Fused validation-tuned": np.full(len(y), np.nan)}
    tuned_predictions = np.full(len(y), -1, dtype=int)
    fold_records = []
    for fold, (train_idx, test_idx) in enumerate(outer_splits, 1):
        train_subjects, test_subjects = set(groups[train_idx]), set(groups[test_idx])
        train_trials, test_trials = set(trials[train_idx]), set(trials[test_idx])
        if train_subjects & test_subjects or train_trials & test_trials:
            raise AssertionError(f"Subject/trial leakage in {setting} fold {fold}")
        x_train, y_train, g_train = x[train_idx], y[train_idx], groups[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]
        inner_splits = min(3, len(np.unique(g_train)))
        if inner_splits < 2:
            raise ValueError("Insufficient training subjects for grouped inner CV")
        inner_cv = GroupKFold(n_splits=inner_splits)
        # Every inner validation partition is subject-disjoint; metrics are scored only
        # on outer test subjects, never on these hyperparameter-selection partitions.
        rf, rf_grid, svm, svm_grid = _models_and_grids(SEED + fold)
        rf_search = GridSearchCV(rf, rf_grid, cv=inner_cv, scoring="roc_auc", n_jobs=1, refit=True)
        svm_search = GridSearchCV(svm, svm_grid, cv=inner_cv, scoring="roc_auc", n_jobs=1, refit=True)
        rf_search.fit(x_train, y_train, groups=g_train)
        svm_search.fit(x_train, y_train, groups=g_train)

        # Validation probabilities are pooled grouped OOF predictions from training
        # subjects only, used to select the tuned fusion rule.
        val_cv = GroupKFold(n_splits=inner_splits)
        p_rf_val = cross_val_predict(rf_search.best_estimator_, x_train, y_train, groups=g_train, cv=val_cv, method="predict_proba", n_jobs=1)[:, 1]
        p_svm_val = cross_val_predict(svm_search.best_estimator_, x_train, y_train, groups=g_train, cv=val_cv, method="predict_proba", n_jobs=1)[:, 1]
        tuned = select_fusion_parameters(p_rf_val, p_svm_val, y_train)

        p_rf, p_svm = rf_search.best_estimator_.predict_proba(x_test)[:, 1], svm_search.best_estimator_.predict_proba(x_test)[:, 1]
        p_equal = 0.5 * p_rf + 0.5 * p_svm
        p_tuned = tuned["weight_rf"] * p_rf + tuned["weight_svm"] * p_svm
        for name, values in (("RF", p_rf), ("SVM", p_svm), ("Fused equal", p_equal), ("Fused validation-tuned", p_tuned)):
            oof[name][test_idx] = values
        tuned_predictions[test_idx] = (p_tuned >= tuned["threshold"]).astype(int)

        fold_record = {
            "dataset": "MobiFall",
            "feature_setting": setting,
            "fold": fold,
            "train_subjects": sorted(train_subjects),
            "test_subjects": sorted(test_subjects),
            "train_class_counts": {"non_fall": int((y_train == 0).sum()), "fall": int((y_train == 1).sum())},
            "test_class_counts": {"non_fall": int((y_test == 0).sum()), "fall": int((y_test == 1).sum())},
            "train_trials": int(len(train_trials)),
            "test_trials": int(len(test_trials)),
            "selected_rf_params": rf_search.best_params_,
            "selected_svm_params": svm_search.best_params_,
            "validation_fusion": tuned,
            "metrics": {
                "RF": binary_metrics(y_test, p_rf),
                "SVM": binary_metrics(y_test, p_svm),
                "Fused equal": binary_metrics(y_test, p_equal),
                "Fused validation-tuned": binary_metrics(y_test, p_tuned, tuned["threshold"]),
            },
        }
        fold_records.append(fold_record)
        fold_path = FOLDS_DIR / f"{setting.lower().replace(' ', '_')}_fold_{fold}.json"
        fold_path.write_text(json.dumps(fold_record, indent=2), encoding="utf-8")
        print(f"{setting} fold {fold}/5: test subjects={len(test_subjects)}, test windows={len(test_idx)}, falls={int(y_test.sum())}, ADL={int((y_test == 0).sum())}")

    pooled = {name: binary_metrics(y, probabilities) for name, probabilities in oof.items()}
    pooled["Fused validation-tuned"] = binary_metrics(y, oof["Fused validation-tuned"], y_pred_override=tuned_predictions)
    fold_summary = {}
    for name in oof:
        metric_names = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "fpr", "fnr", "roc_auc", "pr_auc"]
        fold_summary[name] = {}
        for metric in metric_names:
            values = [fold["metrics"][name][metric] for fold in fold_records if fold["metrics"][name][metric] is not None]
            fold_summary[name][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
    # Subject-cluster bootstrap preserves within-subject window dependence.
    subject_codes, subject_values = pd.factorize(groups, sort=True)
    rng = np.random.default_rng(SEED + 9000)
    ci = {}
    predictions = {name: (prob >= 0.5).astype(int) for name, prob in oof.items()}
    predictions["Fused validation-tuned"] = tuned_predictions.copy()
    for name, pred in predictions.items():
        boot = {metric: [] for metric in ("accuracy", "recall", "f1")}
        for _ in range(2000):
            subject_multiplicity = np.bincount(rng.integers(0, len(subject_values), len(subject_values)), minlength=len(subject_values))
            weights = subject_multiplicity[subject_codes]
            tp = float(np.sum(weights * ((y == 1) & (pred == 1))))
            tn = float(np.sum(weights * ((y == 0) & (pred == 0))))
            fp = float(np.sum(weights * ((y == 0) & (pred == 1))))
            fn = float(np.sum(weights * ((y == 1) & (pred == 0))))
            total = tp + tn + fp + fn
            boot["accuracy"].append((tp + tn) / total)
            boot["recall"].append(tp / (tp + fn) if tp + fn else 0.0)
            boot["f1"].append((2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        ci[name] = {metric: [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))] for metric, vals in boot.items()}
    discordance = {}
    for left, right in (("Fused equal", "SVM"), ("Fused equal", "RF")):
        a, b = predictions[left], predictions[right]
        n01 = int(np.sum((a == y) & (b != y)))
        n10 = int(np.sum((a != y) & (b == y)))
        discordance[f"{left} vs {right}"] = {"fused_only_correct": n01, "other_only_correct": n10,
                                                 "p_value_exact": float(binomtest(min(n01, n10), n01 + n10, .5).pvalue) if n01 + n10 else 1.0}
    pooled["subject_bootstrap_95ci"] = ci
    pooled["mcnemar_exact"] = discordance
    pooled["fold_mean_std"] = fold_summary
    return pooled, fold_records


def _feature_sets(features: pd.DataFrame) -> dict[str, list[str]]:
    cols = [c for c in features.columns if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(features[c])]
    invariant = [
        c for c in cols
        if c.startswith(("acc_mag_", "gyro_mag_", "svmag_")) or c in {"acc_sma", "gyro_sma", "sma"}
    ]
    if not invariant:
        raise ValueError("No orientation-invariant magnitude/SMA features were found")
    paper = ["svmag_mean", "svmag_var", "svmag_p2p", "acc_sma", "gyro_mag_mean", "gyro_mag_p2p"]
    missing = sorted(set(paper) - set(cols))
    if missing:
        raise ValueError(f"Required paper feature columns missing: {missing}")
    return {"all features": cols, "orientation-invariant": invariant, "paper six features": paper}


def _write_metrics_figure(y, pooled_metrics, setting):
    names = ["RF", "SVM", "Fused equal", "Fused validation-tuned"]
    cms = [pooled_metrics[name]["confusion_matrix"] for name in names]
    fig, axes = plt.subplots(1, len(names), figsize=(14, 3.6))
    for ax, name, cm in zip(axes, names, cms):
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), val in np.ndenumerate(matrix):
            ax.text(j, i, str(val), ha="center", va="center")
        ax.set_xticks([0, 1], ["non-fall", "fall"])
        ax.set_yticks([0, 1], ["non-fall", "fall"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle(f"MobiFall pooled OOF confusion matrices — {setting}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"mobifall_confusion_{setting.lower().replace(' ', '_')}.png", dpi=160)
    plt.close(fig)


def _sanity_checks(features: pd.DataFrame, feature_cols: list[str]) -> dict:
    x = features[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)
    y = features.is_fall.to_numpy(dtype=int)
    groups = features.subject_id.to_numpy()
    outer = _checked_group_splits(x, y, groups, N_OUTER_FOLDS)
    shuffled_oof = np.full(len(y), np.nan)
    shuffle_fold_metrics = []
    for repeat in range(5):
        repeat_probabilities = np.full(len(y), np.nan)
        for fold, (train_idx, test_idx) in enumerate(outer, 1):
            rng = np.random.default_rng(SEED + 1000 + repeat * 100 + fold)
            y_shuffle = rng.permutation(y[train_idx])
            model = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=SEED + repeat * 100 + fold, n_jobs=1)
            model.fit(x[train_idx], y_shuffle)
            probabilities = model.predict_proba(x[test_idx])[:, 1]
            repeat_probabilities[test_idx] = probabilities
            shuffle_fold_metrics.append(binary_metrics(y[test_idx], probabilities))
        if repeat == 0:
            shuffled_oof = repeat_probabilities
    shuffle_summary = {}
    for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]:
        values = [row[metric] for row in shuffle_fold_metrics if row[metric] is not None]
        shuffle_summary[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}
    majority = int(np.bincount(y).argmax())
    majority_prob = np.full(len(y), float(majority))
    adl_subjects = set(features.loc[features.is_fall == 0, "subject_id"])
    fall_subjects = set(features.loc[features.is_fall == 1, "subject_id"])
    shared = adl_subjects & fall_subjects
    fall_trials = features[features.is_fall == 1].drop_duplicates("trial_id")
    fall_only_subjects = fall_subjects - adl_subjects
    return {
        "label_shuffle": {
            "method": "Five repeated GroupKFold runs; each RF is trained on randomly permuted training-fold labels and evaluated against untouched held-out labels. Primary scores average fold metrics to avoid pooled-fold calibration shifts.",
            "pooled_oof_first_repeat": binary_metrics(y, shuffled_oof),
            "mean_std_across_folds_and_repeats": shuffle_summary,
            "performance_near_chance": abs(shuffle_summary["roc_auc"]["mean"] - 0.5) < 0.1 and abs(shuffle_summary["balanced_accuracy"]["mean"] - 0.5) < 0.1,
            "chance_reference": 0.5,
        },
        "dataset_identity": {
            "status": "not_applicable",
            "reason": "This experiment uses MobiFall only; no second dataset is included for identity classification.",
        },
        "majority_class_baseline": {
            "predicted_class": "fall" if majority else "non_fall",
            "metrics": binary_metrics(y, majority_prob),
        },
        "subject_trial_leakage": {"status": "asserted absent for every outer fold", "folds": N_OUTER_FOLDS},
        "cohort_composition": {
            "shared_subjects": sorted(shared),
            "fall_only_subjects": sorted(fall_only_subjects),
            "fall_trials_from_shared_subjects": int(fall_trials[fall_trials.subject_id.isin(shared)].trial_id.nunique()),
            "fall_trials_from_fall_only_subjects": int(fall_trials[fall_trials.subject_id.isin(fall_only_subjects)].trial_id.nunique()),
            "interpretation": "ADL trials all come from subjects also represented in the fall set; most fall trials come from the 15 fall-only subjects.",
        },
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FOLDS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    features, cap = build_mobifall_windows(verbose=True)
    manifest_path = write_manifest(features, cap)
    print(f"Window cap={cap}; manifest={manifest_path}")
    print("Fold subject allocation (GroupKFold):")
    groups = features.subject_id.to_numpy()
    y = features.is_fall.to_numpy(dtype=int)
    for i, (_, test_idx) in enumerate(GroupKFold(n_splits=N_OUTER_FOLDS).split(np.zeros((len(y), 1)), y, groups), 1):
        test = features.iloc[test_idx]
        print(f"  fold {i}: subjects={sorted(test.subject_id.unique())}; ADL={int((test.is_fall == 0).sum())}; fall={int(test.is_fall.sum())}")

    results_rows = []
    all_metrics = {}
    all_folds = []
    feature_sets = _feature_sets(features)
    for setting, cols in feature_sets.items():
        print(f"\nRunning {setting}: {len(cols)} features")
        pooled, folds = _metrics_for_settings(features, setting, cols)
        all_metrics[setting] = pooled
        all_folds.extend(folds)
        _write_metrics_figure(y, pooled, setting)
        labels = [
            ("SVM", "SVM"),
            ("RF", "RF"),
            ("Fused equal", "Fused equal weights"),
            ("Fused validation-tuned", "Fused validation-tuned"),
        ]
        for key, label in labels:
            m = pooled[key]
            results_rows.append({"Model": f"{label} ({setting})", "Test set": "MobiFall", "Acc": m["accuracy"], "Prec": m["precision"], "Rec": m["recall"], "F1": m["f1"]})
        print(f"Pooled OOF metrics ({setting}):")
        print(json.dumps(pooled, indent=2))

    sanity = _sanity_checks(features, feature_sets["all features"])
    (REPORTS_DIR / "sanity.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    (REPORTS_DIR / "mobifall_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    (REPORTS_DIR / "mobifall_per_fold.json").write_text(json.dumps(all_folds, indent=2), encoding="utf-8")
    results_path = REPORTS_DIR / "results_table.csv"
    pd.DataFrame(results_rows, columns=["Model", "Test set", "Acc", "Prec", "Rec", "F1"]).to_csv(results_path, index=False)
    print(f"\nSanity checks:\n{json.dumps(sanity, indent=2)}")
    print(f"\nresults_table.csv ({results_path}):")
    print(pd.read_csv(results_path).to_csv(index=False), end="")
    print("\nSaved reports: reports/dataset_manifest.json, reports/mobifall_metrics.json, reports/mobifall_per_fold.json, reports/sanity.json, reports/results_table.csv")


if __name__ == "__main__":
    main()
