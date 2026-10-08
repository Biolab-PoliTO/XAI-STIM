"""
XAI-STIM Signal Processing

Resample MER recordings, split them into overlapping one-second windows, and
apply the preprocessing used to train XAI-STIM: row-wise z-score
normalization, 200-Hz high-pass filtering, and harmonic notch filtering.

Author: Fabrizio Sciscenti
        H-MoVe, Department of Electronics and Telecommunications, Politecnico di Torino
        fabrizio.sciscenti@polito.it
Last updated: 7 October 2026
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "dataset" / "orig"
OUTPUT = ROOT / "dataset" / "processed"
OVERLAPS = ROOT / "results" / "overlap.csv"
SAMPLING_FREQUENCIES = {"LeadPoint": 24000, "NeuroSet": 20000, "AlphaMap": 12000}
TARGET_FS = 10000 # Target sampling frequency for all datasets
WINDOW = 10000 # Number of samples in one-second windows (10 kHz * 1 s)
LOWPASS = 5000 # Low-pass filter cutoff frequency for resampling
HIGHPASS = 200 # High-pass filter cutoff frequency used during model development
NOTCH_FREQUENCIES = np.arange(250, 3000, 50) # 250, 300, ..., 2950 Hz
NOTCH_QUALITY = 50 # Quality factor used during model development
ORDER = 6 # Filter order


def load_matrix(path):
    with np.load(path, allow_pickle=False) as data:
        return data["signals"] if "signals" in data.files else data[data.files[0]]


def resample_signals(matrix, metadata, original_fs):
    """Low-pass and resample a zero-padded recording matrix to 10 kHz."""
    if original_fs <= 0:
        raise ValueError("original_fs must be positive")
    # Match the development importer. For upsampling, a fixed 5-kHz cutoff is
    # at or above the input Nyquist frequency and is therefore omitted.
    lowpass = None
    if original_fs > TARGET_FS:
        lowpass = signal.butter(ORDER, LOWPASS / (original_fs / 2), btype="lowpass")
    rows = []
    metadata = metadata.copy()

    for index, row in enumerate(matrix):
        row = row[: int(metadata.iloc[index]["length"])]
        if not len(row):
            rows.append(np.empty(0, dtype=np.float32))
            continue
        if lowpass is not None:
            row = signal.filtfilt(*lowpass, row)
        rows.append(signal.resample_poly(row, TARGET_FS, original_fs))

    resampled = np.zeros((len(rows), max(map(len, rows), default=0)), dtype=np.float32)
    for index, row in enumerate(rows):
        resampled[index, : len(row)] = row
    metadata["length"] = [len(row) for row in rows]
    return resampled, metadata


def split_signals(matrix, metadata, overlaps):
    """Split a zero-padded 10-kHz recording matrix using patient/class overlaps."""
    overlaps = overlaps.set_index(["patient", "dataset"])
    windows, rows = [], []

    for index, recording in metadata.iterrows():
        class_ = 1 if recording["class"] == 1 else 0
        eta = float(overlaps.loc[(recording["patient"], recording["dataset"]), f"step2_eta{class_}"])
        stride = round(WINDOW * (1 - eta))
        for start in range(0, int(recording["length"]) - WINDOW + 1, stride):
            end = start + WINDOW
            windows.append(matrix[index, start:end])
            row = recording.to_dict()
            row.update({"start": start, "end": end, "overlap": eta})
            rows.append(row)

    columns = [*metadata.columns, "start", "end", "overlap"]
    matrix = np.stack(windows).astype(np.float32) if windows else np.empty((0, WINDOW), np.float32)
    return matrix, pd.DataFrame(rows, columns=columns)


def highpass_filter(matrix):
    """Apply the training-time 200-Hz zero-phase high-pass filter."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != WINDOW:
        raise ValueError(f"Expected a matrix with shape (n, {WINDOW})")
    b, a = signal.butter(ORDER, HIGHPASS, btype="highpass", fs=TARGET_FS)
    return _forward_reverse_filter(matrix, ((b, a),))


def _forward_reverse_filter(matrix, coefficients):
    """Apply the development pipeline's forward/reverse IIR filtering."""
    filtered = np.empty_like(matrix, dtype=np.float32)
    for index, source in enumerate(matrix):
        row = source
        for b, a in coefficients:
            row = signal.lfilter(b, a, row).astype(np.float32)
            row = signal.lfilter(b, a, row[::-1]).astype(np.float32)[::-1]
        filtered[index] = row
    return filtered


def filter_signals(matrix):
    """Apply the high-pass and harmonic notch filters used for training."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != WINDOW:
        raise ValueError(f"Expected a matrix with shape (n, {WINDOW})")
    coefficients = [signal.butter(ORDER, HIGHPASS, btype="highpass", fs=TARGET_FS)]
    coefficients.extend(
        signal.iirnotch(frequency, NOTCH_QUALITY, TARGET_FS)
        for frequency in NOTCH_FREQUENCIES
    )
    return _forward_reverse_filter(matrix, coefficients)


def zscore_normalize(matrix):
    """Apply training-time row-wise amplitude z-score normalization."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[1] != WINDOW:
        raise ValueError(f"Expected a matrix with shape (n, {WINDOW})")
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    return (matrix - mean) / (std + 1e-8)


def process_signals(matrix):
    """Reproduce the complete preprocessing used to train XAI-STIM.

    Normalization intentionally precedes filtering, matching the reference
    ``miscellaneous/utils/processing.py`` implementation.
    """
    return filter_signals(zscore_normalize(matrix)) #zscore_normalize(filter_signals(matrix))


def process_datasets():
    overlaps = pd.read_csv(OVERLAPS)
    for dataset, frequency in SAMPLING_FREQUENCIES.items():
        source = INPUT / dataset
        files = sorted(source.rglob("*.npz"))
        for path in files:
            metadata = pd.read_csv(path.with_suffix(".csv"))
            metadata["dataset"] = dataset
            metadata["class"] = (metadata["class"] == 1).astype(int)
            matrix, metadata = resample_signals(load_matrix(path), metadata, frequency)
            matrix, metadata = split_signals(matrix, metadata, overlaps)
            if len(matrix):
                matrix = process_signals(matrix)

            folder = OUTPUT / dataset / path.relative_to(source).parent
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / f"{path.stem}.npz", signals=matrix)
            metadata.to_csv(folder / f"{path.stem}.csv", index=False)
            print(f"{dataset}/{path.parent.name}/{path.stem}: {matrix.shape}")


if __name__ == "__main__":
    process_datasets()
