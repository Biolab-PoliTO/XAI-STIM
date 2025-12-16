from scipy import signal
import numpy as np
import torch
import sys

def initialize_filter_coefficients(fsamp,
                                   filter_type='highpass',
                                   notch_freqs=None,
                                   lowcut=None,
                                   highcut=None,
                                   order=None):
    """
    Initializes filter coefficients for highpass, lowpass, bandpass, and notch filters.
    Parameters:
        fsamp (float): Sampling frequency of the signal.
        filter_type (str or list): Type(s) of filter(s) to initialize.
        notch_freqs (list or np.ndarray): Frequencies for notch filters.
        lowcut (float): Low cutoff frequency for highpass/bandpass.
        highcut (float): High cutoff frequency for lowpass/bandpass.
        order (int): Filter order.
    Returns:
        tuple: Two dictionaries containing the filter coefficients (b, a).
    """
    if order is None:
        order = 6
    if lowcut is None:
        lowcut = 200
    if highcut is None:
        highcut = 3000
    if notch_freqs is None:
        notch_freqs = np.arange(100, 4000, 50)
    if isinstance(filter_type, str):
        filter_types = [filter_type]
    else:
        filter_types = filter_type

    b = {}
    a = {}
    nyquist = 0.5 * fsamp

    for ftype in filter_types:
        if ftype == 'highpass':
            b['high'], a['high'] = signal.butter(order, lowcut / nyquist, btype='highpass')
        elif ftype == 'lowpass':
            b['low'], a['low'] = signal.butter(order, highcut / nyquist, btype='lowpass')
        elif ftype == 'bandpass':
            b['band'], a['band'] = signal.butter(order, [lowcut / nyquist, highcut / nyquist], btype='bandpass')
        elif ftype == 'notch':
            for freq in notch_freqs:
                b[f'notch_{freq}Hz'], a[f'notch_{freq}Hz'] = signal.iirnotch(freq, 50, fsamp)
        else:
            raise ValueError(f"Unknown filter type: {ftype}")

    return b, a

def process_data(data, 
                 fsamp,
                 normalization='zscore',
                 filter_type=['bandpass', 'notch'],
                 notch_freqs=None,
                 lowcut=None,
                 highcut=None,
                 order=None,
                 b=None,
                 a=None,
                 verbose=True):
    """
    Applies normalization and a series of filters to the input data.
    Parameters:
        data (numpy.ndarray or torch.Tensor): Input MER.
        fsamp (float): Sampling frequency.
        normalization (str): 'zscore', 'minmax', or 'none'.
        filter_type (str or list): Filter types to apply.
        notch_freqs, lowcut, highcut, order: Filter parameters.
    Returns:
        numpy.ndarray or torch.Tensor: Filtered MER.
    """
    # Convert torch.Tensor to numpy if needed
    isTensor = isinstance(data, torch.Tensor)
    if isTensor:
        data = data.cpu().numpy().squeeze()

    if data.ndim == 1:
        data = data[np.newaxis, :]

    # Apply normalization
    if verbose:
        print("Applying normalization...")
    if normalization == 'zscore':
        data = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-8)
    elif normalization == 'minmax':
        data = 2 * (data - np.min(data, axis=1, keepdims=True)) / (np.max(data, axis=1, keepdims=True) - np.min(data, axis=1, keepdims=True) + 1e-8) - 1
    elif normalization == 'mediqr':
        median = np.median(data, axis=1, keepdims=True)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=1, keepdims=True))
        data = 2 * (data - median) / (iqr + 1e-8) - 1
        # data = np.tanh(data)
    elif normalization == 'percentile':
        p1 = np.percentile(data, 1, axis=1, keepdims=True)
        p99 = np.percentile(data, 99, axis=1, keepdims=True)
        data = 2 * (data - p1) / (p99 - p1 + 1e-8) - 1
        # data = np.tanh(data)

    # Initialize filter coefficients
    if filter_type is not None:
        if b is None:
            b, a = initialize_filter_coefficients(
                fsamp,
                filter_type=filter_type,
                notch_freqs=notch_freqs,
                lowcut=lowcut,
                highcut=highcut,
                order=order
            )
        # Apply filtering to each row (channel/segment) of the input data
        filtered_data = np.empty_like(data, dtype=np.float32)
        total = data.shape[0]
        for i in range(total):
            row = data[i]
            for key in b:
                row = signal.lfilter(b[key], a[key], row).astype(np.float32)
            filtered_data[i] = row
            if verbose:
                # Print progress
                progress = f"\rFiltering data: {i+1}/{total} rows"
                sys.stdout.write(progress)
                sys.stdout.flush()
    else:
        filtered_data = data.copy()

    if verbose:
        print()  # Move to next line after completion

    # Convert back to torch.Tensor if needed
    if isTensor:
        filtered_data = torch.tensor(filtered_data, dtype=torch.float32)

    return filtered_data