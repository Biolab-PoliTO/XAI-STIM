"""
One-dimensional Grad-CAM for XAI-STIM.

Signals are resampled to 10 kHz when necessary, divided into non-overlapping
one-second windows, preprocessed with the data-preparation functions, and
mapped at the self-attention layer, as in the development inference script.
Each window map is min-max scaled; maps are then concatenated per signal,
smoothed with a 500-sample Gaussian, and scaled again to the [0, 1] range.

Author: Fabrizio Sciscenti
        H-MoVe, Department of Electronics and Telecommunications, Politecnico di Torino
        fabrizio.sciscenti@polito.it
Last updated: 7 October 2026
"""

import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "01_data_preparation"))
sys.path.insert(0, str(HERE))
import processing  # noqa: E402
from inference import MODELS, load_model, preprocess  # noqa: E402

WINDOW = processing.WINDOW
TARGET_FS = processing.TARGET_FS
SMOOTHING_SIGMA = 500


def _matrix(signals):
    matrix = np.asarray(signals)
    vector = matrix.ndim == 1
    if vector:
        matrix = matrix[None, :]
    if matrix.ndim != 2:
        raise ValueError("signals must be a vector or a matrix with signals on rows")
    if matrix.shape[1] == 0:
        raise ValueError("signals cannot be empty")
    return matrix, vector


def _resample(matrix, sampling_frequency):
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive")
    if sampling_frequency == TARGET_FS:
        return np.asarray(matrix, dtype=np.float32)
    metadata = pd.DataFrame({"length": [matrix.shape[1]] * len(matrix)})
    resampled, _ = processing.resample_signals(matrix, metadata, sampling_frequency)
    return resampled


def _windows(row):
    if len(row) <= WINDOW:
        return np.asarray([np.pad(row, (0, WINDOW - len(row)))]), [len(row)]

    number_of_full_windows, remainder = divmod(len(row), WINDOW)
    windows = [
        row[index * WINDOW : (index + 1) * WINDOW]
        for index in range(number_of_full_windows)
    ]
    lengths = [WINDOW] * number_of_full_windows
    if remainder:
        # Match development: explain the last complete second, then retain its
        # non-overlapping tail when concatenating the maps.
        windows.append(row[-WINDOW:])
        lengths.append(remainder)
    return np.asarray(windows), lengths


def _window_gradcam(model, windows, device, target_class):
    """Apply the reference GradCAM1D algorithm to each window."""
    maps = []
    probabilities = []
    activations = None
    gradients = None

    def forward_hook(_module, _inputs, output):
        nonlocal activations
        activations = output[0] if isinstance(output, tuple) else output

    def backward_hook(_module, _grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0] if isinstance(grad_output, tuple) else grad_output

    forward_handle = model.attention.register_forward_hook(forward_hook)
    backward_handle = model.attention.register_full_backward_hook(backward_hook)
    try:
        # The development inference calls GradCAM1D once per window. Retaining
        # that behavior also makes its global min-max normalization window-wise.
        for window in windows:
            input_tensor = torch.from_numpy(window[None, :]).float().to(device)
            input_tensor.requires_grad_(True)
            logits, _ = model(input_tensor)
            probabilities.append(torch.softmax(logits, dim=1)[0, 1].item())
            if target_class is None:
                classes = logits.argmax(dim=1)
            else:
                if target_class not in (0, 1):
                    raise ValueError("target_class must be 0, 1, or None")
                classes = torch.tensor([target_class], device=device)

            model.zero_grad()
            one_hot = torch.zeros_like(logits)
            one_hot[torch.arange(len(logits), device=device), classes] = 1
            logits.backward(gradient=one_hot, retain_graph=True)

            if activations is None or gradients is None:
                raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")

            features = activations
            feature_gradients = gradients
            if features.dim() == 3:
                features = features.squeeze(-1)
            if feature_gradients.dim() == 3:
                feature_gradients = feature_gradients.squeeze(-1)
            features = features.reshape(1, model.num_patches, -1)
            feature_gradients = feature_gradients.reshape(1, model.num_patches, -1)

            weights = feature_gradients.mean(dim=1, keepdim=True)
            cam = torch.relu((weights * features).sum(dim=2)).unsqueeze(1)
            cam = F.interpolate(cam, size=WINDOW, mode="nearest")
            minimum, maximum = cam.min(), cam.max()
            if minimum == maximum:
                cam = torch.zeros_like(cam)
            else:
                cam = (cam - minimum) / (maximum - minimum)
            maps.append(cam.detach().cpu().numpy().squeeze())
    finally:
        forward_handle.remove()
        backward_handle.remove()
    return np.stack(maps), np.asarray(probabilities, dtype=np.float32)


def gradcam1D(
    signals,
    sampling_frequency,
    model_path=MODELS / "all_in_one.pt",
    target_class=1,
    device=None,
    return_probabilities=False,
):
    """Compute a smoothed sample-wise Grad-CAM relevance map for each signal.

    Parameters
    ----------
    signals : array-like, shape (samples,) or (n_signals, samples)
        Signals sampled at ``sampling_frequency``.
    sampling_frequency : float
        Input sampling frequency in hertz. Inputs are resampled to 10 kHz.
    model_path : path-like
        XAI-STIM state dictionary to explain.
    target_class : {0, 1, None}
        Explained class. ``None`` explains each window's predicted class.
    return_probabilities : bool
        If true, also return the mean class-1 probability across the windows
        belonging to each signal.

    Returns
    -------
    numpy.ndarray
        Maps aligned to the resampled signals. A vector input returns a vector.
        When ``return_probabilities`` is true, returns ``(maps, probabilities)``.
    """
    matrix, vector = _matrix(signals)
    matrix = _resample(matrix, sampling_frequency)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(model_path, device)

    relevance = []
    signal_probabilities = []
    for row in matrix:
        windows, lengths = _windows(row)
        windows = preprocess(windows)
        maps, probabilities = _window_gradcam(model, windows, device, target_class)
        signal_probabilities.append(float(probabilities.mean()))
        pieces = []
        for index, (cam, length) in enumerate(zip(maps, lengths)):
            overlapping_tail = index == len(maps) - 1 and length < WINDOW and len(maps) > 1
            pieces.append(cam[-length:] if overlapping_tail else cam[:length])
        concatenated = np.concatenate(pieces)
        smoothed = gaussian_filter1d(concatenated, sigma=SMOOTHING_SIGMA)
        minimum, maximum = smoothed.min(), smoothed.max()
        if maximum > minimum:
            smoothed = (smoothed - minimum) / (maximum - minimum)
        else:
            smoothed = np.zeros_like(smoothed)
        relevance.append(smoothed)

    relevance = np.asarray(relevance, dtype=np.float32)
    probabilities = np.asarray(signal_probabilities, dtype=np.float32)
    if vector:
        relevance = relevance[0]
        probabilities = float(probabilities[0])
    return (relevance, probabilities) if return_probabilities else relevance


def plot_gradcam(
    signals,
    relevance_maps,
    probabilities,
    sampling_frequency=TARGET_FS,
    metadata=None,
    output_path=None,
    show=False,
):
    """Plot equally spaced signals, probability-scaled maps, and class-1 scores.

    Physical depths are used to sort and label the rows, but not to position
    them. For display only, each non-zero relevance map is normalized and then
    multiplied by its signal's class-1 probability, so its maximum colour
    value equals the corresponding probability bar.
    """
    signals, _ = _matrix(signals)
    relevance_maps, _ = _matrix(relevance_maps)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if signals.shape != relevance_maps.shape:
        raise ValueError("signals and relevance_maps must have the same shape")
    if len(probabilities) != len(signals):
        raise ValueError("One probability is required for each signal")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must contain finite values in the [0, 1] range")
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive")

    if metadata is None:
        depth_labels = np.arange(len(signals))
        depth_label = "Signal"
        order = np.arange(len(signals))
    else:
        metadata = metadata.reset_index(drop=True)
        if len(metadata) != len(signals) or "depth" not in metadata:
            raise ValueError("metadata must contain one depth value per signal")
        depths = pd.to_numeric(metadata["depth"], errors="raise").to_numpy(float) / 1000
        depth_label = "Depth (mm)"
        order = np.argsort(depths, kind="stable")
        depth_labels = depths[order]

    signals = signals[order]
    relevance_maps = relevance_maps[order]
    probabilities = probabilities[order]
    if metadata is None:
        depth_labels = depth_labels[order]
    row_positions = np.arange(len(signals), dtype=float)
    duration = signals.shape[1] / sampling_frequency
    time = np.arange(signals.shape[1]) / sampling_frequency
    band_height = 0.82

    scale = np.percentile(np.abs(signals), 99)

    figure, (signal_axis, probability_axis) = plt.subplots(
        1,
        2,
        figsize=(15, max(6, 0.62 * len(signals))),
        sharey=True,
        layout="constrained",
        gridspec_kw={"width_ratios": (4.5, 1.1), "wspace": 0.02},
    )
    for index, (signal_row, relevance, probability, position) in enumerate(
        zip(signals, relevance_maps, probabilities, row_positions)
    ):
        relevance_maximum = np.nanmax(relevance)
        displayed_relevance = (
            relevance / relevance_maximum * probability
            if np.isfinite(relevance_maximum) and relevance_maximum > 0
            else np.zeros_like(relevance)
        )
        signal_axis.imshow(
            displayed_relevance[None, :],
            extent=(0, duration, position + 0.5, position - 0.5),
            aspect="auto",
            cmap="jet",
            vmin=0,
            vmax=1,
            alpha=0.52,
            interpolation="nearest",
        )
        centered = signal_row - np.median(signal_row)
        # scale = np.percentile(np.abs(centered), 99)
        trace = (
            position + centered / scale * band_height * 0.38
            if scale
            else np.full_like(centered, position)
        )
        signal_axis.plot(time, trace, color="black", linewidth=0.35, rasterized=True)

    colors = np.where(probabilities >= 0.5, "#4daf4a", "#ef3038")
    probability_axis.barh(
        row_positions, probabilities, height=band_height * 0.72, color=colors
    )
    probability_axis.axvline(0.5, color="black", linestyle="--", linewidth=1.2)
    for position, probability in zip(row_positions, probabilities):
        probability_axis.text(
            min(probability + 0.02, 1.04),
            position,
            f"{probability:.2f}",
            va="center",
            fontsize=9,
        )

    signal_axis.set_xlim(0, duration)
    signal_axis.set_xlabel("Time (s)", fontweight="bold")
    signal_axis.set_ylabel(depth_label, fontweight="bold")
    signal_axis.set_yticks(row_positions)
    signal_axis.set_yticklabels([f"{value:g}" for value in depth_labels])
    signal_axis.grid(axis="x", alpha=0.22)
    probability_axis.set_xlim(0, 1.1)
    probability_axis.set_xlabel("IN-STN probability", fontweight="bold")
    probability_axis.set_xticks((0, 0.5, 1))
    probability_axis.grid(axis="x", alpha=0.3)
    signal_axis.set_ylim(len(signals) - 0.5, -0.5)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return figure


def main():
    parser = argparse.ArgumentParser(description="Compute and plot Grad-CAM for sample_data.npz")
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds to plot")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--show", action="store_true", help="Display the figure after saving it")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "sample_gradcam.png"
    )
    arguments = parser.parse_args()
    if arguments.duration <= 0:
        parser.error("--duration must be positive")

    data_path = ROOT / "sample_data.npz"
    metadata_path = ROOT / "sample_meta.csv"
    matrix = processing.load_matrix(data_path)
    metadata = pd.read_csv(metadata_path, dtype={"patient": str})
    if len(matrix) != len(metadata):
        raise ValueError("sample_data.npz and sample_meta.csv have different row counts")
    datasets = metadata["dataset"].dropna().unique()
    if len(datasets) != 1 or datasets[0] not in processing.SAMPLING_FREQUENCIES:
        raise ValueError("Sample metadata must contain one dataset with a known sampling frequency")
    sampling_frequency = processing.SAMPLING_FREQUENCIES[datasets[0]]
    samples = round(arguments.duration * sampling_frequency)
    if (metadata["length"] < samples).any():
        raise ValueError("At least one sample signal is shorter than the requested duration")

    matrix = matrix[:, :samples]
    resampled = _resample(matrix, sampling_frequency)
    relevance, probabilities = gradcam1D(
        resampled,
        TARGET_FS,
        model_path=MODELS / "all_in_one.pt",
        target_class=1,
        device=arguments.device,
        return_probabilities=True,
    )
    figure = plot_gradcam(
        resampled,
        relevance,
        probabilities,
        TARGET_FS,
        metadata,
        arguments.output,
        arguments.show,
    )
    if not arguments.show:
        plt.close(figure)
    print(f"Saved Grad-CAM figure to {arguments.output}")


compute_gradcam = gradcam1D


if __name__ == "__main__":
    main()
