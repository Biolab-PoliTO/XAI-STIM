import cv2
import sys
import numpy as np
import scipy
import matplotlib
from pyts.image import GramianAngularField
import pywt
import torch
from torchview import draw_graph

def data_to_spectrogram(tbl: np.ndarray, sample_rate, nperseg, window, overlap, onesided, 
                        NFFT=None, to_image=False, target_size=None, verbose=True):  
    """
    ...
    """
    
    # Initialize final tensor
    if target_size is not None:
        ret = np.zeros((tbl.shape[0], target_size[0], target_size[1]), dtype=np.float32)
    else:
        _, _, spectrogram = scipy.signal.spectrogram(tbl[0, :], sample_rate, nperseg=nperseg, return_onesided=onesided, noverlap=np.round(overlap * nperseg))
        ret = np.zeros((tbl.shape[0], spectrogram.shape[0], spectrogram.shape[1]), dtype=np.float32)
    if to_image:
        ret = np.zeros((tbl.shape[0], 3, ret.shape[1], ret.shape[2]), dtype=np.float32)
        cmap = matplotlib.colormaps.get_cmap('viridis')

    # Spectrogram computation
    for i in range(tbl.shape[0]):      
        _, _, spectrogram = scipy.signal.spectrogram(tbl[i, :]-np.mean(tbl[i, :]), sample_rate, nperseg=nperseg,
                                               window=window, noverlap=np.round(overlap * nperseg), nfft=NFFT, detrend='linear',
                                               return_onesided=onesided, scaling='density', axis=-1, mode='psd'
                                               )
        if target_size is not None:
            spectrogram = cv2.resize(spectrogram, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
        if not onesided:
            spectrogram = np.fft.fftshift(spectrogram, axes=0)
        if to_image:
            spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram) + 1e-8)
            spectrogram_rgb = cmap(spectrogram)[:, :, :3]
            spectrogram_rgb = (spectrogram_rgb * 255).astype(np.uint8)
            ret[i, :, :, :] = np.transpose(spectrogram_rgb, (2, 0, 1))
        else:
            ret[i, :, :] = spectrogram
        
        if verbose: print(f"\rTo spectrogram {i+1}/{tbl.shape[0]}", end='', flush=True)
    if verbose: print()
    return ret

def data_to_scalogram(tbl: np.ndarray, sample_rate, wavelet, scales=None,
                         to_image=False, target_size=(100, 100), verbose=True) -> np.ndarray:
    """
    ...
    """

    n_signals, _ = tbl.shape
    n_scales, n_time = target_size

    if scales is None:
        scales = np.linspace(1, 128, n_scales)

    # Initialize output array
    scalograms = np.zeros((n_signals, n_scales, n_time), dtype=np.float32)
    if to_image:
        scalograms = np.zeros((n_signals, 3, n_scales, n_time), dtype=np.uint8)
        cmap = matplotlib.colormaps.get_cmap('viridis')

    for i in range(n_signals):
        rec = tbl[i]
        coeffs, _ = pywt.cwt(rec, scales, wavelet, sampling_period=1/sample_rate)
        coeffs_mag = np.abs(coeffs)

        # Resample along time axis
        coeffs_resampled = scipy.signal.resample(coeffs_mag, n_time, axis=1)
        if to_image:
            coeffs_resampled = (coeffs_resampled - np.min(coeffs_resampled)) / (np.max(coeffs_resampled) - np.min(coeffs_resampled) + 1e-8)
            coeffs_rgb = cmap(coeffs_resampled)[:, :, :3]
            coeffs_rgb = (coeffs_rgb * 255).astype(np.uint8)
            scalograms[i, :, :, :] = np.transpose(coeffs_rgb, (2, 0, 1)) # channel-first
        else:
            scalograms[i] = coeffs_resampled
        if verbose: print(f"\rTo CWT {i+1}/{tbl.shape[0]}", end='', flush=True)
    if verbose: print()

    return scalograms

def data_to_GASF(tbl: np.ndarray, to_image=False, target_size=100, verbose=True):
    """
    ...
    """

    # Initialize final tensor
    ret = np.zeros((tbl.shape[0], target_size, target_size), dtype=np.float32)
    if to_image:
        ret = np.zeros((tbl.shape[0], 3, ret.shape[1], ret.shape[2]), dtype=np.uint8)
        cmap = matplotlib.colormaps.get_cmap('viridis')

    # GASF computation
    gasf_transform = GramianAngularField(method='summation', overlapping=True, image_size=target_size)
    for i in range(tbl.shape[0]):
        gasf_matrix = gasf_transform.fit_transform(tbl[i, :].reshape(1, -1))[0]
        if to_image:
            gasf_matrix = (gasf_matrix - np.min(gasf_matrix)) / (np.max(gasf_matrix) - np.min(gasf_matrix) + 1e-8)
            gasf_rgb = cmap(gasf_matrix)[:, :, :3]
            gasf_rgb = (gasf_rgb * 255).astype(np.uint8)
            ret[i, :, :, :] = np.transpose(gasf_rgb, (2, 0, 1))
        else:
            ret[i, :, :] = gasf_matrix
        if verbose: print(f"\rTo GASF {i+1}/{tbl.shape[0]}", end='', flush=True)
    if verbose: print()
    return ret

def remove_artifacts_with_replacement(data, fsamp):
    L = data.shape[0]
    params = {'w': int(100e-3 * fsamp), 'threshold': 1.33}
    artMask = np.zeros(L, dtype=bool)
    Wbis = 0
    t = 0
    start = 0
    stop = params['w']
    pvar = 1e14
    first_it = True
    s = params['w'] // 2
    w = params['w']
    clearSignal = np.zeros(L, dtype=data.dtype)
    # varMask = np.zeros(L, dtype=data.dtype)
    latest_stable_epoch = np.zeros(params['w'], dtype=data.dtype)
    while start < L:
        epoch = data[start:stop]
        nvar = np.var(epoch)
        # varMask[start:stop] = np.ones(stop-start) * (nvar / pvar)
        if nvar >= params['threshold'] * pvar:
            W = [1,1]
            # replacement
            clearSignal[start:stop] = latest_stable_epoch.copy()[:stop-start] ####
        else:
            W = [0,0]
            pvar = nvar
            clearSignal[start:stop] = epoch.copy()
            latest_stable_epoch = epoch.copy()
        if first_it:
            Wbis = W[0]
            first_it = False

        if stop - start < s:
            s = stop - start

        cls = np.mean([W[0], Wbis])
        artMask[t:t + s] = cls >= 0.5

        Wbis = W[1]

        t += s
        start += s
        stop = start + w
        if stop > L:
            stop = L
            w = stop - start

    return clearSignal, artMask

def remove_artifacts_with_replacement_v2(data, fsamp):
    L = data.shape[0]
    params = {'w': int(100e-3 * fsamp), 'threshold': 1.33}
    artMask = np.zeros(L, dtype=bool)
    Wbis = 0
    t = 0
    start = 0
    stop = params['w']
    pvar = 1e14
    first_it = True
    s = params['w'] // 2
    w = params['w']
    clearSignal = np.zeros(L, dtype=data.dtype)
    #varMask = np.zeros(L, dtype=data.dtype)
    latest_stable_epoch = np.zeros(params['w'], dtype=data.dtype)
    while start < L:

        epoch = data[start:stop]
        nvar = np.var(epoch)
        #varMask[start:stop] = np.ones(stop-start) * (nvar / pvar)
        if nvar >= params['threshold'] * pvar:
            W = [1,1]
            clearSignal[start:stop] = latest_stable_epoch.copy()[:stop-start] ####
        else:
            W = [0,0]
            pvar = nvar
            clearSignal[start:stop] = epoch.copy()
            latest_stable_epoch = epoch.copy()
        
        if first_it:
            Wbis = W[0]
            first_it = False

        if stop - start < s:
            s = stop - start

        cls = np.mean([W[0], Wbis])
        artMask[t:t + s] = cls >= 0.5

        Wbis = W[1]

        t += s
        start += s
        stop = start + w
        if stop > L:
            stop = L
            w = stop - start

    artMask_diff = np.diff(artMask.astype(int), prepend=0)
    fall_edges = np.where(artMask_diff == -1)[0]
    rise_edges = np.where(artMask_diff == 1)[0]

    for j in fall_edges:
        i = np.max(rise_edges[rise_edges<j])
        if (j-i)/s > 1:
            n = (((j-i)/s)//2).astype(int)
            start_idx = j-n*s
            donor = clearSignal[j:j+s]
            for k in range(n):
                clearSignal[start_idx+k*s:start_idx+(k+1)*s] = donor

    return clearSignal, artMask

def get_var_mask(data, fsamp):
    L = data.shape[0]
    params = {'w': int(100e-3 * fsamp), 'threshold': 1.33}
    start = 0
    stop = params['w']
    pvar = 1e14
    varMask = np.zeros(L, dtype=data.dtype)
    s = params['w'] // 2
    w = params['w']
    while start < L:
        epoch = data[start:stop]
        nvar = np.var(epoch)
        varMask[start:stop] = np.ones(stop-start) * (nvar / pvar)
        if nvar < params['threshold'] * pvar:
            pvar = nvar
        start += s
        stop = start + w
        if stop > L:
            stop = L

    return varMask

def data_to_features(tbl: np.ndarray, fsamp, verbose=True, features_extremes=None):

    def extract_features(segment, fsamp):
    #     max_vals = [8.707813,0.016264,0.073682,0.133387,0.16357,2769,0.435164,0.936993,2.680869]
    #     min_vals = [1.357211,0.002564,0.019996,-0.250695,0.041139,1644.88,0.153334,0.833393,-0.096321]

        # PSD calculation
        freqs, Pxx = scipy.signal.welch(segment, fsamp, nperseg=fsamp // 10)
        P_tot = np.sum(Pxx)
        PSD = Pxx / P_tot
        freqsA, PxxA = scipy.signal.welch(np.abs(segment), fsamp, nperseg=fsamp // 10)
        P_totA = np.sum(PxxA)
        PSDA = PxxA / P_totA
    
        # Feature extraction
        avgAbsDiff = np.mean(np.abs(segment - np.mean(segment)))
        pr_8_13Hz = np.sum(PSDA[(freqsA >= 8) & (freqsA <= 13)])
        pr_30_70Hz = np.sum(PSDA[(freqsA >= 30) & (freqsA <= 70)])
        skewness = scipy.stats.skew(segment)
        PSDratio = np.sum(Pxx[(freqs >= 3000) & (freqs <= 4000)]) / np.sum(Pxx[(freqs >= 1600) & (freqs <= 2200)])
        zc_count = ((segment[:-1] * segment[1:]) < 0).sum()
        pr_1_2kHz = np.sum(PSD[(freqs >= 1000) & (freqs <= 2000)])
        PSDindex = 0
        for j in range(22):
            PSDindex += np.sum(PSD[(freqs >= 255 + j * 100) & (freqs <= 245 + (j + 1) * 100)])   
        kurt = scipy.stats.kurtosis(segment)

        features = np.array([avgAbsDiff, pr_8_13Hz, pr_30_70Hz, skewness, PSDratio, zc_count, pr_1_2kHz, PSDindex, kurt]).reshape(1, -1)
        return features

    features = np.zeros((tbl.shape[0], 9), dtype=np.float32)
    for i in range(tbl.shape[0]):
        segment = tbl[i]
        clear_segment, _ = remove_artifacts_with_replacement_v2(segment, fsamp)
        features[i] = extract_features(clear_segment, fsamp)
        if verbose: print(f"\rTo features {i+1}/{tbl.shape[0]}", end='', flush=True)
    
    if verbose: print("\nApplying min-max scaling...")
    if features_extremes is None:
        max_vals = np.percentile(features, 99, axis=0)
        min_vals = np.percentile(features, 1, axis=0)
        features_extremes = np.concatenate((max_vals.reshape(1, -1), min_vals.reshape(1, -1)), axis=0)
    else:
        max_vals = features_extremes[0,:]
        min_vals = features_extremes[1,:]
    for i in range(features.shape[1]):
        features[:, i] = (features[:, i] - min_vals[i]) / (max_vals[i] - min_vals[i])
    features = np.clip(features, 0, 1)

    if verbose: print()

    return features, features_extremes

def remove_artifacts_fun(tbl, fsamp, verbose = True):
    clear_data = np.zeros((tbl.shape[0], tbl.shape[1]), dtype=np.float32)
    for i in range(tbl.shape[0]):
        segment = tbl[i]
        clear_segment, _ = remove_artifacts_with_replacement_v2(segment, fsamp)
        clear_data[i] = clear_segment
        if verbose: print(f"\rRemoved artifacts from {i+1}/{tbl.shape[0]}", end='', flush=True)

    if verbose: print()

    return clear_data


def transform_data(data, fsamp, model, verbose=True, features_extremes=None):
    """
    ...
    """

    if isinstance(model, str):
        model_name = model
    else:
        model_name = model.__class__.__name__

    # Convert torch.Tensor to numpy if needed
    isTensor = isinstance(data, torch.Tensor)
    if isTensor:
        data = data.cpu().numpy().squeeze()

    data_transform = []

    if model_name == 'ResCNN_SpAT':
        data_transform = data_to_spectrogram(data, fsamp, 
                                            nperseg=128, 
                                            window='hann', 
                                            overlap=0.50,
                                            onesided=False,
                                            to_image=True,
                                            #target_size=(153, 153),
                                            target_size = (100, 100),
                                            verbose=verbose)
    elif model_name == 'CNN_ReSpAT':
        data_transform = data_to_spectrogram(data, fsamp, 
                                            nperseg=128, 
                                            window='hann', 
                                            overlap=0.50,
                                            onesided=False,
                                            to_image=True,
                                            #target_size=(153, 153),
                                            target_size = (100, 100),
                                            verbose=verbose)
        
    elif model_name == 'CNN_CBAM':
        CWT_data = data_to_scalogram(data, fsamp,
                                    wavelet='cmor3-3', 
                                    to_image=True, 
                                    target_size=(100, 100),
                                    verbose=verbose)
        GASF_data = data_to_GASF(data,
                                to_image=True,
                                target_size=100,
                                verbose=verbose)
        data_transform = np.concatenate([CWT_data, GASF_data], axis=1)

    elif model_name == 'ResNet_AT': 
        data_transform = data_to_spectrogram(data, fsamp, 
                                            nperseg=256, 
                                            window='tukey', 
                                            overlap=0.25, 
                                            onesided=True,
                                            to_image=False,
                                            verbose=verbose)
        
    elif model_name == 'ML_STIM':
        data_transform, features_extremes = data_to_features(data, fsamp,
                                          verbose=verbose,
                                          features_extremes=features_extremes)
    elif model_name == 'CNN_PwMhAT_ML_STIM':
        data_transform = remove_artifacts_fun(data, fsamp, verbose=verbose)

    else:
        data_transform = data.copy()

    # Convert back to torch.Tensor if needed
    if isTensor:
        data_transform = torch.tensor(data_transform, dtype=torch.float32)

    return data_transform, features_extremes

if __name__ == "__main__":

    import import_network
    import apply_processing

    # Generate a sample 1-second signal (10000 samples at 10kHz)
    fsamp = 10000
    
    processing_opts = {'fsamp': fsamp, 
                       #'normalization': None, #'zscore',
                       'filter_type': ['highpass','notch'],
                       'notch_freqs': np.arange(100, 4000, 50),
                       'lowcut':250,
                       'order': 6}

    signal_matrix = np.load('utils/sample_data/out_in_mers.npz')
    data = signal_matrix['data']

    data_filt = apply_processing.process_data(data, normalization=None, **processing_opts)
    data_norm = apply_processing.process_data(data, normalization='zscore', **processing_opts)

    model_name = 'CNN_ReSpAT'
    model, _, _, _, _, _ = import_network.import_model(model_name)

    # clearSignal, artMask = remove_artifacts_with_replacement_v2(data[1], fsamp)

    data_filt_spec, _ = transform_data(data_filt, fsamp, model)
    data_norm_spec, _ = transform_data(data_norm, fsamp, model)
    # data_transform_2, _ = transform_data(data, fsamp, model, features_extremes=features_extremes)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Filtered Signal Spectrogram')
    plt.imshow(np.transpose(data_filt_spec[1, :, :, :], (1, 2, 0))/255, origin='lower')
    plt.subplot(1, 2, 2)
    plt.title('Normalized Signal Spectrogram')
    plt.imshow(np.transpose(data_norm_spec[1, :, :, :], (1, 2, 0))/255, origin='lower')

    print('Ciao')