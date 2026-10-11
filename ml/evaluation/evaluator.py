"""
FallGuard AI - ML Evaluation & Visualization Engine
Computes comprehensive healthcare metrics (Accuracy, Precision, Recall, F1, Fall Recall, FPR, FNR, ROC-AUC)
and generates high-resolution figures.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Headless backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
from typing import Dict, List, Any, Optional

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_classifier(
    model, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    class_names: List[str],
    fall_class_name: str = "FALL"
) -> Dict[str, Any]:
    """
    Evaluate a trained model against test data with multi-class and fall-specific metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    acc = float(accuracy_score(y_test, y_pred))
    macro_prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    macro_rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    
    # Fall-specific sensitivity and specificity
    fall_idx = class_names.index(fall_class_name) if fall_class_name in class_names else -1
    if fall_idx != -1:
        y_test_fall = (y_test == fall_idx).astype(int)
        y_pred_fall = (y_pred == fall_idx).astype(int)
        
        # Binary confusion matrix for Fall: [[TN, FP], [FN, TP]]
        cm_fall = confusion_matrix(y_test_fall, y_pred_fall, labels=[0, 1])
        if cm_fall.shape == (2, 2):
            tn, fp, fn, tp = cm_fall.ravel()
            fall_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            fall_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            fall_fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            fall_fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
            fall_f1 = float(2 * (fall_precision * fall_recall) / (fall_precision + fall_recall)) if (fall_precision + fall_recall) > 0 else 0.0
        else:
            fall_recall = 1.0
            fall_precision = 1.0
            fall_fpr = 0.0
            fall_fnr = 0.0
            fall_f1 = 1.0
            
        # Fall ROC-AUC
        if y_prob is not None and len(np.unique(y_test_fall)) > 1:
            fall_prob = y_prob[:, fall_idx]
            fall_roc_auc = float(roc_auc_score(y_test_fall, fall_prob))
        else:
            fall_roc_auc = 1.0
    else:
        fall_recall = macro_rec
        fall_precision = macro_prec
        fall_fpr = 0.0
        fall_fnr = 0.0
        fall_f1 = macro_f1
        fall_roc_auc = 1.0

    cm_full = confusion_matrix(y_test, y_pred, labels=range(len(class_names))).tolist()
    cls_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    return {
        "accuracy": acc,
        "macro_precision": macro_prec,
        "weighted_precision": weighted_prec,
        "macro_recall": macro_rec,
        "weighted_recall": weighted_rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "fall_recall": fall_recall,
        "fall_precision": fall_precision,
        "fall_f1": fall_f1,
        "fall_fpr": fall_fpr,
        "fall_fnr": fall_fnr,
        "fall_roc_auc": fall_roc_auc,
        "confusion_matrix": cm_full,
        "classification_report": cls_report,
        "y_pred": y_pred,
        "y_prob": y_prob
    }

def plot_confusion_matrix(cm: List[List[int]], class_names: List[str], title: str, save_path: Path):
    """Generate high-contrast normalized confusion matrix heatmap."""
    plt.figure(figsize=(9, 7))
    cm_arr = np.array(cm)
    cm_norm = cm_arr.astype('float') / (cm_arr.sum(axis=1)[:, np.newaxis] + 1e-12)
    
    # Custom annotations showing count & percentage
    annot_matrix = np.empty_like(cm_arr, dtype=object)
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            annot_matrix[i, j] = f"{cm_arr[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
            
    sns.heatmap(cm_norm, annot=annot_matrix, fmt="", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Accuracy'})
    plt.title(f"Confusion Matrix - {title}", fontsize=14, pad=15, fontweight="bold")
    plt.xlabel("Predicted Activity", fontsize=12, labelpad=10)
    plt.ylabel("Ground Truth Activity", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_model_comparison(comparison_df: pd.DataFrame, save_path: Path):
    """Plot bar chart comparing Random Forest, SVM, and hybrid fusion metrics."""
    plt.figure(figsize=(12, 6))
    
    metrics = ["Accuracy", "Macro F1", "Fall Recall", "Fall Precision"]
    df_plot = comparison_df.melt(id_vars=["Model"], value_vars=metrics, var_name="Metric", value_name="Score")
    
    palette = ["#2563eb", "#059669", "#dc2626", "#d97706"]
    ax = sns.barplot(data=df_plot, x="Model", y="Score", hue="Metric", palette=palette)
    plt.title("Model Performance Comparison (Healthcare & Activity Metrics)", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Score (0.0 - 1.0)", fontsize=12)
    plt.xlabel("Model Architecture", fontsize=12)
    plt.ylim(0.0, 1.08)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(title="Metric", loc="lower right", framealpha=0.95)
    
    # Add score annotations above bars
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height*100:.1f}%",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, rotation=90, xytext=(0, 3),
                        textcoords='offset points')
                        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_feature_importance(importance_dict: Dict[str, float], top_n: int = 20, save_path: Path = None):
    """Plot top N influential kinematic and frequency features."""
    if not importance_dict:
        return
        
    sorted_feats = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [x[0] for x in sorted_feats][::-1]
    values = [x[1] for x in sorted_feats][::-1]
    
    plt.figure(figsize=(10, 8))
    bars = plt.barh(names, values, color="#0284c7")
    plt.title(f"Top {top_n} Most Influential Sensor Features", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Gini Feature Importance / Weight", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    
    # Add value annotations
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, f"{width:.4f}", 
                 va='center', ha='left', fontsize=9, color="#1e293b")
                 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_activity_distribution(df_features: pd.DataFrame, save_path: Path):
    """Plot activity and fall class distribution in training and test sets."""
    plt.figure(figsize=(10, 5))
    order = df_features["activity"].value_counts().index
    ax = sns.countplot(data=df_features, x="activity", order=order, hue="activity", palette="viridis", legend=False)
    plt.title("Sensor Dataset Activity & Fall Distribution", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Activity Category", fontsize=12)
    plt.ylabel("Window Count (2.56s Windows)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{int(height)}",
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                    textcoords='offset points')
                    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
