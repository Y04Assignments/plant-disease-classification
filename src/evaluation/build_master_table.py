"""Merge each model's saved final-results CSV into one comparison table.

Run from the repository root:  python src/evaluation/build_master_table.py
Models whose results file does not exist yet (e.g. EfficientNetB0) are listed as pending.
"""
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

# Each model team saved slightly different column names; map them onto one schema.
SOURCES = {
    "Custom CNN": ("customcnn_final_results.csv", {
        "accuracy": "accuracy", "weighted_precision": "weighted_precision", "weighted_recall": "weighted_recall",
        "weighted_f1": "weighted_f1", "weighted_roc_auc": "weighted_roc_auc", "total_params": "total_params",
        "trainable_params": "trainable_params", "training_time_seconds": "training_time_seconds",
        "model_size_mb": "model_size_mb", "cpu_inference_latency_ms": "cpu_inference_latency_ms"}),
    "MobileNetV2": ("mobilenetv2_final_results.csv", {
        "accuracy": "accuracy", "weighted_precision": "weighted_precision", "weighted_recall": "weighted_recall",
        "weighted_f1": "weighted_f1", "weighted_roc_auc": "weighted_roc_auc",
        "training_time_seconds": "training_time_seconds"}),
    "ResNet50": ("resnet50_final_results.csv", {
        "test_accuracy": "accuracy", "weighted_precision": "weighted_precision", "weighted_recall": "weighted_recall",
        "weighted_f1": "weighted_f1", "roc_auc_weighted": "weighted_roc_auc", "total_params": "total_params",
        "finetune_trainable_params": "trainable_params", "total_training_time_sec": "training_time_seconds"}),
    "EfficientNetB0": ("efficientnetb0_final_results.csv", {
        "accuracy": "accuracy", "weighted_precision": "weighted_precision", "weighted_recall": "weighted_recall",
        "weighted_f1": "weighted_f1", "weighted_roc_auc": "weighted_roc_auc", "total_params": "total_params",
        "trainable_params": "trainable_params", "training_time_seconds": "training_time_seconds"}),
}
# Values reported in a notebook's printed output but not saved to its results CSV.
NOTEBOOK_VALUES = {
    "MobileNetV2": {"total_params": 2261827},  # 02_Preprocessing.ipynb, "Total parameters" cell
}
COLUMNS = ["accuracy", "weighted_precision", "weighted_recall", "weighted_f1", "weighted_roc_auc",
           "total_params", "trainable_params", "training_time_seconds", "model_size_mb", "cpu_inference_latency_ms"]


def build():
    rows = []
    for model, (filename, mapping) in SOURCES.items():
        path = RESULTS_DIR / filename
        row = {"model": model, "status": "pending", "source_file": filename}
        if path.exists():
            src = pd.read_csv(path).iloc[0]
            row["status"] = "evaluated"
            row.update({target: src[col] for col, target in mapping.items() if col in src})
            row.update(NOTEBOOK_VALUES.get(model, {}))
        rows.append(row)
    table = pd.DataFrame(rows).reindex(columns=["model", "status", *COLUMNS, "source_file"])
    table.to_csv(RESULTS_DIR / "master_comparison_draft.csv", index=False)
    return table


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(build().to_string(index=False))
