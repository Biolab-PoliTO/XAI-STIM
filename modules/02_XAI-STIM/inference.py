"""
XAI-STIM inference utilities and validation-set export.

Calling ``inference`` on a complete 10-kHz recording splits it into one-second
windows with configurable overlap, then applies the same row-wise z-score
normalization, 200-Hz high-pass filter, and harmonic notch filters used during
model development. When this file is run as a script, it evaluates the supplied
AiO and LODO weights on their corresponding processed datasets.

Author: Fabrizio Sciscenti
        H-MoVe, Department of Electronics and Telecommunications, Politecnico di Torino
        fabrizio.sciscenti@polito.it
Last updated: 11 October 2026
"""

import argparse
import sys
import time
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
METADATA_COLUMNS = [
    "patient", "dataset", "electrode", "depth", "class", "start", "end",
]
OUTPUT_COLUMNS = [*METADATA_COLUMNS, "softmax0", "softmax1", "time"]

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


def preprocess(signals, coefficients=None):
    """Apply the complete preprocessing to signals of any sample length."""
    if coefficients is None:
        coefficients = processing.create_filter_coefficients()
    return processing.process_signals(signals, coefficients)


def window_recording(signal, overlap=0.0):
    """Split one recording into complete one-second windows.

    Parameters
    ----------
    signal : array-like, shape (n_samples,)
        A single recording sampled at 10 kHz.
    overlap : float, default=0.5
        Fraction shared by consecutive windows. Must satisfy ``0 <= overlap < 1``.

    Returns
    -------
    windows : numpy.ndarray, shape (n_windows, 10000)
        Complete one-second windows. A trailing incomplete window is discarded.
    starts : numpy.ndarray, shape (n_windows,)
        Start sample of every window in the original recording.
    """
    recording = np.asarray(signal)
    if recording.ndim != 1:
        raise ValueError("signal must be a single one-dimensional recording")
    if len(recording) < WINDOW:
        raise ValueError(f"signal must contain at least {WINDOW} samples (one second)")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")

    stride = max(1, round(WINDOW * (1 - overlap)))
    starts = np.arange(0, len(recording) - WINDOW + 1, stride, dtype=int)
    windows = np.stack([recording[start : start + WINDOW] for start in starts])
    return windows, starts


def _predict_batches(matrix, model, device, batch_size, coefficients=None):
    """Predict batches and return probabilities and per-segment elapsed times.

    If ``coefficients`` are supplied, every batch is preprocessed before
    inference. Filter construction is deliberately excluded from the timed
    region. Without coefficients, the matrix is assumed to be preprocessed.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[1] != WINDOW:
        raise ValueError(f"Expected data with shape (n, {WINDOW})")
    probabilities = []
    elapsed_per_segment = []
    with torch.inference_mode():
        for start in range(0, len(matrix), batch_size):
            source = matrix[start : start + batch_size]
            started = time.time()
            if coefficients is not None:
                source = preprocess(source, coefficients)
            batch = torch.from_numpy(np.asarray(source, dtype=np.float32)).to(device)
            logits, _ = model(batch)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            per_segment = (time.time() - started) / len(source)
            elapsed_per_segment.extend([per_segment] * len(source))
    if not probabilities:
        return np.empty((0, 2), np.float32), np.empty(0, dtype=float)
    return np.concatenate(probabilities), np.asarray(elapsed_per_segment)


def _predict_processed(matrix, model, device, batch_size):
    return _predict_batches(matrix, model, device, batch_size)


def inference(
    signals,
    model_path=MODELS / "all_in_one.pt",
    batch_size=256,
    device=None,
    return_times=False,
    overlap=0.0,
    return_window_starts=False,
):
    """Return probabilities for one recording or a matrix of prepared windows.

    A one-dimensional input is treated as a complete 10-kHz recording and is
    divided into one-second windows using ``overlap`` (50% by default). A
    two-dimensional input remains supported for evaluation code and must
    contain one prepared one-second window per row. Set ``return_times=True``
    to return the preprocessing-plus-inference time per window. Set
    ``return_window_starts=True`` to also return each window's start sample;
    this option is only valid for a one-dimensional recording.
    """
    source = np.asarray(signals)
    recording = source.ndim == 1
    if recording:
        matrix, starts = window_recording(source, overlap)
    else:
        matrix = source
        starts = None
        if return_window_starts:
            raise ValueError("return_window_starts requires a one-dimensional recording")

    device = _device(device)
    model = load_model(model_path, device)
    coefficients = processing.create_filter_coefficients()
    probabilities, times = _predict_batches(
        matrix, model, device, batch_size, coefficients
    )

    outputs = [probabilities]
    if return_times:
        outputs.append(times)
    if return_window_starts:
        outputs.append(starts)
    return tuple(outputs) if len(outputs) > 1 else probabilities


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
    """Write window probabilities and per-segment batch time in seconds."""
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
        probabilities, times = _predict_processed(matrix, model, device, batch_size)
        missing = [column for column in METADATA_COLUMNS if column not in metadata]
        if missing:
            raise ValueError(f"Missing metadata columns in {path.with_suffix('.csv')}: {missing}")
        table = metadata[METADATA_COLUMNS].copy()
        table["softmax0"] = probabilities[:, 0]
        table["softmax1"] = probabilities[:, 1]
        table["time"] = times
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
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    arguments = parser.parse_args()
    run_all_validations(arguments.batch_size, arguments.device)


if __name__ == "__main__":
    main()
