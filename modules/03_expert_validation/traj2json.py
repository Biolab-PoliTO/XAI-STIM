"""Export MER trajectories for the XAI-STIM expert-validation interface.

The JSON contains display-rate signals, probability-scaled Grad-CAM maps, and
one mean class-1 probability per recording/depth.

Author: Fabrizio Sciscenti
        H-MoVe, Department of Electronics and Telecommunications, Politecnico di Torino
        fabrizio.sciscenti@polito.it

Last updated: 8 October 2026
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREPARATION = HERE.parent / "01_data_preparation"
XAI = HERE.parent / "02_XAI-STIM"

# Numbered and hyphenated module folders are not importable package names.
sys.path.insert(0, str(PREPARATION))
sys.path.insert(0, str(XAI))
import processing  # noqa: E402
from gradcam1D import gradcam1D  # noqa: E402
from inference import MODELS  # noqa: E402


DEFAULT_OUTPUT_FS = 8000
REQUIRED_METADATA = {"depth", "length", "class"}


def _load_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path, dtype={"patient": str})
    if len(metadata.columns) == 1:
        metadata = pd.read_csv(path, sep=";", dtype={"patient": str})
    missing = sorted(REQUIRED_METADATA.difference(metadata.columns))
    if missing:
        raise ValueError(f"Missing metadata columns in {path}: {missing}")
    return metadata


def _input_frequencies(metadata: pd.DataFrame, input_sampling_frequency: int | None) -> np.ndarray:
    if input_sampling_frequency is not None:
        if input_sampling_frequency <= 0:
            raise ValueError("input_sampling_frequency must be positive")
        return np.full(len(metadata), int(input_sampling_frequency), dtype=int)

    if "dataset" not in metadata:
        raise ValueError(
            "Metadata must contain 'dataset', or input_sampling_frequency must be provided"
        )
    unknown = sorted(
        set(metadata["dataset"].dropna()).difference(processing.SAMPLING_FREQUENCIES)
    )
    if unknown or metadata["dataset"].isna().any():
        raise ValueError(
            "Cannot infer input sampling frequency for dataset values: "
            f"{unknown or ['<missing>']}; provide input_sampling_frequency"
        )
    return metadata["dataset"].map(processing.SAMPLING_FREQUENCIES).to_numpy(int)


def _display_resample(values: np.ndarray, output_sampling_frequency: int) -> np.ndarray:
    """Downsample a 10-kHz result for JSON display without duplicating preprocessing."""
    divisor = gcd(processing.TARGET_FS, output_sampling_frequency)
    result = signal.resample_poly(
        np.asarray(values),
        output_sampling_frequency // divisor,
        processing.TARGET_FS // divisor,
    )
    return np.asarray(result, dtype=np.float32)


def process_recordings(
    matrix: np.ndarray,
    metadata: pd.DataFrame,
    *,
    input_sampling_frequency: int | None = None,
    output_sampling_frequency: int = DEFAULT_OUTPUT_FS,
    model_path: str | Path = MODELS / "all_in_one.pt",
    device: str | None = None,
) -> tuple[list[list[float]], list[list[float]], list[float]]:
    """Prepare recordings and compute interface-ready signals, maps, and scores.

    Rows are trimmed according to the metadata ``length`` field. Dataset names
    select the acquisition rate from the data-preparation module unless one
    common rate is supplied explicitly. Internally, existing pipeline calls
    perform the 10-kHz resampling, model preprocessing, inference, and Grad-CAM.
    The final light-weight resampling only reduces JSON/display size.
    """
    matrix = np.asarray(matrix)
    metadata = metadata.reset_index(drop=True).copy()
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional with recordings on rows")
    if len(matrix) != len(metadata):
        raise ValueError("Signal and metadata row counts differ")
    if not len(matrix):
        raise ValueError("At least one recording is required")
    if not 0 < output_sampling_frequency <= processing.TARGET_FS:
        raise ValueError(
            f"output_sampling_frequency must be in [1, {processing.TARGET_FS}] Hz"
        )

    lengths = pd.to_numeric(metadata["length"], errors="raise").to_numpy(int)
    if np.any(lengths <= 0) or np.any(lengths > matrix.shape[1]):
        raise ValueError("Every metadata length must be between 1 and the matrix width")
    frequencies = _input_frequencies(metadata, input_sampling_frequency)

    # Grouping avoids repeatedly loading the model when recordings share their
    # acquisition frequency and duration, while still supporting mixed inputs.
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(zip(frequencies, lengths)):
        groups[(int(key[0]), int(key[1]))].append(index)

    signals_10k: list[np.ndarray | None] = [None] * len(matrix)
    relevance_10k: list[np.ndarray | None] = [None] * len(matrix)
    probabilities = np.empty(len(matrix), dtype=np.float32)

    for (frequency, length), indices in groups.items():
        source = np.asarray(matrix[indices, :length])
        group_metadata = pd.DataFrame({"length": [length] * len(indices)})
        resampled, resampled_metadata = processing.resample_signals(
            source, group_metadata, frequency
        )
        valid_length = int(resampled_metadata.iloc[0]["length"])
        resampled = resampled[:, :valid_length]
        maps, scores = gradcam1D(
            resampled,
            processing.TARGET_FS,
            model_path=model_path,
            target_class=1,
            device=device,
            return_probabilities=True,
        )
        if maps.ndim == 1:
            maps = maps[None, :]
            scores = np.asarray([scores], dtype=np.float32)
        for local_index, source_index in enumerate(indices):
            signals_10k[source_index] = resampled[local_index]
            relevance_10k[source_index] = maps[local_index]
            probabilities[source_index] = scores[local_index]

    display_signals = [
        _display_resample(row, output_sampling_frequency) for row in signals_10k
    ]
    display_relevance = []
    for row, probability in zip(relevance_10k, probabilities):
        relevance = np.clip(
            _display_resample(row, output_sampling_frequency), 0.0, None
        )
        maximum = float(relevance.max())
        if maximum > 0:
            relevance = relevance / maximum * probability
        else:
            relevance = np.zeros_like(relevance)
        display_relevance.append(relevance)

    # Plotly expects a rectangular trajectory. Only genuinely shorter rows are
    # padded; their inference and relevance were computed before padding.
    width = max(map(len, display_signals))
    display_signals = [np.pad(row, (0, width - len(row))) for row in display_signals]
    display_relevance = [np.pad(row, (0, width - len(row))) for row in display_relevance]
    return (
        [row.astype(float).tolist() for row in display_signals],
        [row.astype(float).tolist() for row in display_relevance],
        probabilities.astype(float).tolist(),
    )


def trajectory_to_json(
    matrix_path: str | Path,
    metadata_path: str | Path,
    output_path: str | Path,
    *,
    output_sampling_frequency: int = DEFAULT_OUTPUT_FS,
    input_sampling_frequency: int | None = None,
    model_path: str | Path = MODELS / "all_in_one.pt",
    device: str | None = None,
) -> Path:
    """Convert one NPZ/CSV trajectory pair to the validation-tool JSON format."""
    matrix_path = Path(matrix_path)
    metadata_path = Path(metadata_path)
    output_path = Path(output_path)
    matrix = processing.load_matrix(matrix_path)
    metadata = _load_metadata(metadata_path)
    signals, relevance, probabilities = process_recordings(
        matrix,
        metadata,
        input_sampling_frequency=input_sampling_frequency,
        output_sampling_frequency=output_sampling_frequency,
        model_path=model_path,
        device=device,
    )

    labels = pd.to_numeric(metadata["class"], errors="raise").astype(int).tolist()
    depths = (pd.to_numeric(metadata["depth"], errors="raise") / 1000).tolist()
    patient = str(metadata.iloc[0].get("patient", "unknown"))
    side = str(metadata.iloc[0].get("side", "unknown"))
    identifier = matrix_path.stem
    trajectory = {
        "id": identifier,
        "name": f"Trajectory {identifier} - Patient {patient} ({side})",
        "sampling_rate": int(output_sampling_frequency),
        "saliency_method": "gradcam1D",
        "labels": labels,
        "signals": signals,
        "depths": depths,
        "channel_probabilities": probabilities,
        "saliency": relevance,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(trajectory, stream, separators=(",", ":"), allow_nan=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an MER trajectory to XAI-STIM expert-validation JSON"
    )
    parser.add_argument("matrix", nargs="?", type=Path, default=ROOT / "sample_data.npz")
    parser.add_argument("metadata", nargs="?", type=Path, default=ROOT / "sample_meta.csv")
    parser.add_argument("--output", type=Path, help="Default: <script folder>/<matrix stem>.json")
    parser.add_argument(
        "--output-fs",
        type=int,
        default=DEFAULT_OUTPUT_FS,
        help=f"JSON sampling frequency in Hz (default: {DEFAULT_OUTPUT_FS})",
    )
    parser.add_argument(
        "--input-fs",
        type=int,
        help="Input frequency in Hz; otherwise inferred from metadata dataset",
    )
    parser.add_argument("--model", type=Path, default=MODELS / "all_in_one.pt")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    arguments = parser.parse_args()
    output = arguments.output or HERE / f"{arguments.matrix.stem}.json"
    result = trajectory_to_json(
        arguments.matrix,
        arguments.metadata,
        output,
        output_sampling_frequency=arguments.output_fs,
        input_sampling_frequency=arguments.input_fs,
        model_path=arguments.model,
        device=arguments.device,
    )
    print(f"Saved validation trajectory to {result}")


if __name__ == "__main__":
    main()
