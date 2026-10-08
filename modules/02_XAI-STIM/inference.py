"""
XAI-STIM inference utilities and validation-set export.

Calling ``inference`` on raw one-second signals applies the same row-wise
z-score normalization, 200-Hz high-pass filter, and harmonic notch filters
used during model development. When this file is run as a script, it evaluates
the supplied AiO and LODO weights on their corresponding processed datasets.

Author: Fabrizio Sciscenti
        H-MoVe, Department of Electronics and Telecommunications, Politecnico di Torino
        fabrizio.sciscenti@polito.it
Last updated: 7 October 2026
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROCESSED = ROOT / "dataset" / "processed"
MODELS = HERE / "trained_models"
RESULTS = ROOT / "results"
WINDOW = 10000
OUTPUT_COLUMNS = [
    "patient", "dataset", "electrode", "depth", "class", "start", "end",
    "softmax0", "softmax1",
]

# The preparation folder starts with a number, so it cannot be imported with a
# normal dotted package name. Adding the folder itself keeps one authoritative
# implementation of preprocessing.
sys.path.insert(0, str(HERE.parent / "01_data_preparation"))
sys.path.insert(0, str(HERE))
import processing  # noqa: E402
from xai_stim import XAI_STIM  # noqa: E402


def _device(device=None):
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def load_model(model_path=MODELS / "all_in_one.pt", device=None):
    """Load one of the distributed XAI-STIM state dictionaries."""
    device = _device(device)
    model = XAI_STIM()
    state = torch.load(Path(model_path), map_location=device, weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model.to(device).eval()


def preprocess(signals):
    """Apply the complete preprocessing used during model development."""
    matrix = np.asarray(signals)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[1] != WINDOW:
        raise ValueError(f"Expected a signal or row-wise matrix with {WINDOW} samples")
    return processing.process_signals(matrix)


def _predict_processed(matrix, model, device, batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != WINDOW:
        raise ValueError(f"Expected processed data with shape (n, {WINDOW})")
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(matrix), batch_size):
            batch = torch.from_numpy(matrix[start : start + batch_size]).to(device)
            logits, _ = model(batch)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities) if probabilities else np.empty((0, 2), np.float32)


def inference(
    signals,
    model_path=MODELS / "all_in_one.pt",
    batch_size=256,
    device=None,
):
    """Return class probabilities for a signal or a row-wise signal matrix.

    Each input row must be a one-second signal sampled at 10 kHz. A vector
    input returns two probabilities; a matrix returns shape ``(n_signals, 2)``.
    """
    vector = np.asarray(signals).ndim == 1
    device = _device(device)
    model = load_model(model_path, device)
    probabilities = _predict_processed(preprocess(signals), model, device, batch_size)
    return probabilities[0] if vector else probabilities


predict_proba = inference


def _evaluation_files(dataset=None):
    if dataset is None:
        return sorted((PROCESSED / "test").rglob("*.npz"))
    return sorted(
        path
        for split in ("train", "val", "test")
        for path in (PROCESSED / split / dataset).rglob("*.npz")
    )


def export_validation(model_path, output_path, dataset=None, batch_size=256, device=None):
    """Run one validation phase and write its window-level probability table."""
    device = _device(device)
    model = load_model(model_path, device)
    tables = []
    files = _evaluation_files(dataset)
    if not files:
        scope = "test set" if dataset is None else f"complete {dataset} dataset"
        raise FileNotFoundError(f"No processed NPZ files found for the {scope}")

    for index, path in enumerate(files, 1):
        matrix = processing.load_matrix(path)
        metadata = pd.read_csv(path.with_suffix(".csv"), dtype={"patient": str})
        if len(matrix) != len(metadata):
            raise ValueError(f"Signal/metadata row mismatch for {path}")
        probabilities = _predict_processed(matrix, model, device, batch_size)
        missing = [column for column in OUTPUT_COLUMNS[:-2] if column not in metadata]
        if missing:
            raise ValueError(f"Missing metadata columns in {path.with_suffix('.csv')}: {missing}")
        table = metadata[OUTPUT_COLUMNS[:-2]].copy()
        table["softmax0"] = probabilities[:, 0]
        table["softmax1"] = probabilities[:, 1]
        tables.append(table)
        print(f"\r{Path(model_path).stem}: {index}/{len(files)} files", end="", flush=True)

    result = pd.concat(tables, ignore_index=True)[OUTPUT_COLUMNS]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"\nSaved {len(result)} predictions to {output_path}")
    return result


def run_all_validations(batch_size=256, device=None):
    """Export AiO test predictions and all three complete LODO datasets."""
    phases = (
        ("all_in_one.pt", "all_in_one.csv", None),
        ("leave_AlphaMap_out.pt", "leave_AlphaMap_out.csv", "AlphaMap"),
        ("leave_LeadPoint_out.pt", "leave_LeadPoint_out.csv", "LeadPoint"),
        ("leave_NeuroSet_out.pt", "leave_NeuroSet_out.csv", "NeuroSet"),
    )
    return {
        checkpoint: export_validation(
            MODELS / checkpoint, RESULTS / filename, dataset, batch_size, device
        )
        for checkpoint, filename, dataset in phases
    }


def main():
    parser = argparse.ArgumentParser(description="Run all XAI-STIM validation inferences")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    arguments = parser.parse_args()
    run_all_validations(arguments.batch_size, arguments.device)


if __name__ == "__main__":
    main()
