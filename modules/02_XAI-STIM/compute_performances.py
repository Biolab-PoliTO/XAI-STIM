"""
Compute overall and patient-wise XAI-STIM classification performances.

By default, this script evaluates ``all_in_one.csv`` and every
``leave_*_out.csv`` prediction file in the project ``results`` folder. Each
output is saved beside its input as ``<prediction_stem>_performance.csv``.

Author: Fabrizio Sciscenti
        H-MoVe, Department of Electronics and Telecommunications, Politecnico di Torino
        fabrizio.sciscenti@polito.it
Last updated: 8 October 2026
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
REQUIRED_COLUMNS = {"patient", "dataset", "class", "softmax1"}
OUTPUT_COLUMNS = [
    "scope",
    "patient",
    "dataset",
    "n_samples",
    "auc",
    "f1_score",
    "sensitivity",
    "specificity",
]
THRESHOLD = 0.5


def compute_metrics(table):
    """Compute binary metrics from all rows in a prediction table."""
    labels = table["class"].to_numpy(dtype=int)
    probabilities = table["softmax1"].to_numpy(dtype=float)
    predictions = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    auc = roc_auc_score(labels, probabilities) if np.unique(labels).size == 2 else np.nan

    return {
        "n_samples": len(table),
        "auc": auc,
        "f1_score": f1_score(labels, predictions, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def evaluate_predictions(table):
    """Return one overall row followed by one row per patient and dataset."""
    missing = sorted(REQUIRED_COLUMNS.difference(table.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if table.empty:
        raise ValueError("The prediction table is empty")
    if not table["class"].isin((0, 1)).all():
        raise ValueError("The class column must contain only 0 and 1")
    probabilities = pd.to_numeric(table["softmax1"], errors="coerce")
    if probabilities.isna().any() or not np.isfinite(probabilities).all():
        raise ValueError("The softmax1 column contains missing or non-finite values")
    if not probabilities.between(0, 1).all():
        raise ValueError("The softmax1 column must contain probabilities in [0, 1]")

    rows = [
        {
            "scope": "overall",
            "patient": "ALL",
            "dataset": "ALL",
            **compute_metrics(table),
        }
    ]
    for (patient, dataset), group in table.groupby(
        ["patient", "dataset"], sort=True, dropna=False
    ):
        rows.append(
            {
                "scope": "patient",
                "patient": patient,
                "dataset": dataset,
                **compute_metrics(group),
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def evaluate_file(path):
    """Evaluate one prediction CSV and save its corresponding performance CSV."""
    path = Path(path)
    table = pd.read_csv(path, dtype={"patient": str, "dataset": str})
    performances = evaluate_predictions(table)
    output = path.with_name(f"{path.stem}_performance.csv")
    performances.to_csv(output, index=False)
    print(f"Saved {len(performances)} performance rows to {output}")
    return performances


def prediction_files():
    """Find the AiO and LODO prediction files produced by inference.py."""
    files = []
    all_in_one = RESULTS / "all_in_one.csv"
    if all_in_one.exists():
        files.append(all_in_one)
    files.extend(sorted(RESULTS.glob("leave_*_out.csv")))
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Compute overall and patient-wise XAI-STIM performances"
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Prediction CSV files; defaults to results/all_in_one.csv and leave_*_out.csv",
    )
    arguments = parser.parse_args()
    files = arguments.files or prediction_files()
    if not files:
        parser.error(f"No prediction CSV files found in {RESULTS}")
    for path in files:
        evaluate_file(path)


if __name__ == "__main__":
    main()
