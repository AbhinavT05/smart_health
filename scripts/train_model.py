"""
FallGuard AI - Hybrid RF + SVM training CLI.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from data.download_datasets import build_dataset_pipeline
from ml.training.trainer import train_and_evaluate_all_models


def main():
    print("\n" + "=" * 75)
    print("      FALLGUARD AI - HYBRID RF + SVM TRAINING")
    print("=" * 75 + "\n")

    processed_csv = BASE_DIR / "data" / "processed" / "unified_sensor_stream.csv"
    if not processed_csv.exists():
        print("[1/3] Dataset missing. Running download/ingest pipeline...")
        build_dataset_pipeline()
        if not processed_csv.exists():
            print(
                "ERROR: Dataset still missing after ingest. "
                "Training was not performed and no metrics were invented."
            )
            sys.exit(1)
    else:
        print(f"[1/3] Using compiled sensor dataset: {processed_csv}")

    print("\n[2/3] Training Random Forest and calibrated SVM, then selecting fusion on validation...")
    bundle = train_and_evaluate_all_models(dataset_csv_path=processed_csv, sampling_rate=50.0)

    metrics = bundle["metrics"]
    fusion = bundle["fusion"]
    print("\n" + "=" * 75)
    print("                      HYBRID TRAINING COMPLETE")
    print("=" * 75)
    print(f"  * Architecture:              {bundle['model_name']}")
    print(f"  * Fusion:                    {fusion['formula']}")
    print(f"  * Weights (validation only): w_RF={fusion['weight_rf']}  w_SVM={fusion['weight_svm']}")
    print(f"  * Decision threshold:        {fusion['threshold']}")
    print(f"  * Held-out Fall Recall:      {metrics['fall_recall'] * 100:.2f}%")
    print(f"  * Held-out Fall Precision:   {metrics['fall_precision'] * 100:.2f}%")
    print(f"  * Held-out Fall F1:          {metrics['fall_f1'] * 100:.2f}%")
    if metrics.get("fall_roc_auc") is not None:
        print(f"  * Held-out Fall ROC-AUC:     {metrics['fall_roc_auc'] * 100:.2f}%")
    print(f"  * Artifacts:                 data/models/fall_detection_model.joblib")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
