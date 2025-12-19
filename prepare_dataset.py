"""
MicroElectrode Recordings (MERs) Dataset Preparation for Subthalamic Nucleus (STN) Classification
=================================================================================================

This script processes MicroElectrode Recordings (MER) from multiple acquisition systems
and prepares them for classification as inside-STN (class 1) vs outside-STN (class 0).

Main steps:
1. Import raw datasets from three sources (AlphaMap, NeuroSet, NeuroOmega)
2. Compute optimal overlap for window extraction (variable overlap per recording)
3. Extract fixed-length segments with computed overlap
4. Generate balanced test/construction splits with cross-validation folds

The overlap computation ensures balanced representation across classes and datasets.

-----------------------------------------------------------------------------------------------
Author: Fabrizio Sciscenti
        Ph.D. Student,
        BIOLAB, Department of Electronics and Telecommunications (DET), Politecnico di Torino
        https://biolab.polito.it/people/fabrizio-sciscenti/
        fabrizio.sciscenti@polito.it

Last Updated: 17/12/2025
-----------------------------------------------------------------------------------------------
"""

import os
import numpy as np
import pandas as pd
from scipy import signal
from collections import defaultdict


def import_dataset(base_path, dataset_names, fsamp, allowed_electrodes=None):
    """
    Import and preprocess MER datasets from multiple acquisition systems.
    
    Applies low-pass filtering and resampling to standardize signals across datasets.
    Only imports recordings with allowed electrode labels.
    
    Parameters:
    -----------
    base_path : str
        Root directory containing dataset folders
    dataset_names : list of str
        Names of datasets to import (e.g., ['AlphaMap', 'NeuroSet', 'NeuroOmega'])
    fsamp : dict
        Sampling frequencies for each dataset {dataset_name: frequency}
    allowed_electrodes : list of str, optional
        List of allowed electrode labels (default: ['central', 'anterior', 'posterior', 'medial', 'lateral'])
    
    Returns:
    --------
    X : list of np.ndarray
        List of preprocessed signals
    meta_df : pd.DataFrame
        Metadata for each recording (patient, dataset, side, depth, class, electrode, length)
    """
    all_data = []
    all_meta = []
    f_new = 10000  # Target sampling frequency
    f_c = 5000     # Cutoff frequency for low-pass filter
    
    # Set default allowed electrodes if not provided
    if allowed_electrodes is None:
        allowed_electrodes = ['Central', 'Anterior', 'Posterior', 'Medial', 'Lateral', 'Electrode1']
    
    print(f"Allowed electrodes: {allowed_electrodes}")
    skipped_count = 0
    imported_count = 0

    for dataset_name in dataset_names:
        print(f"\nImporting dataset: {dataset_name}")

        fs = fsamp[dataset_name]
        b, a = signal.butter(6, [f_c / (fs / 2)], btype='lowpass')
        dataset_path = os.path.join(base_path, dataset_name)
        
        if not os.path.isdir(dataset_path):
            print(f"Warning: Dataset path not found: {dataset_path}")
            continue
            
        patient_list = os.listdir(dataset_path)
        for num, pxx in enumerate(patient_list):
            print(f"\rProcessing patient: {pxx} ({num+1}/{len(patient_list)})", end='', flush=True)

            pxx_path = os.path.join(dataset_path, pxx)
            if not os.path.isdir(pxx_path):
                continue
                
            for fname in os.listdir(pxx_path):
                if not fname.endswith('.npy') and not fname.endswith('.npz'):
                    continue
                    
                # Load signal
                fpath = os.path.join(pxx_path, fname)
                data = np.load(fpath, allow_pickle=True)
                data = data[data.files[0]] if fname.endswith('.npz') else data

                # Load metadata
                # Expected format: patient, side, electrode, depth, length, class
                if fname.endswith('.npz'):
                    mpath = fname.replace('.npz', '.csv')
                else:
                    mpath = fname.replace('.npy', '.csv')
                if not os.path.exists(os.path.join(pxx_path, mpath)):
                    print(f"Warning: Metadata file not found: {os.path.join(pxx_path, mpath)}")
                    meta = pd.DataFrame()
                else:
                    meta = pd.read_csv(os.path.join(pxx_path, mpath))
                    if len(meta.columns) == 1:
                        meta = pd.read_csv(os.path.join(pxx_path, mpath), sep=';')
                
                # Apply filtering and resampling

                for i, x in enumerate(data):
                    x = x[:meta['length'].iloc[i] if 'length' in meta.columns else len(x)]  # Trim to original length
                    
                    # Extract electrode information
                    electrode = meta['electrode'].iloc[i] if 'electrode' in meta.columns else 'unknown'
                    
                    # Filter by allowed electrodes
                    if electrode not in allowed_electrodes or len(x) < 1000:
                        skipped_count += 1
                        continue
                    
                    filtered = signal.filtfilt(b, a, x)
                    resampled = signal.resample_poly(filtered, f_new, fs)
                    all_data.append(resampled)

                    class_val = int(meta['class'].iloc[i]) if 'class' in meta.columns else -1
                    if class_val > 1: class_val = 0
                    
                    all_meta.append({
                        'patient': meta['patient'].iloc[i] if 'patient' in meta.columns else fname.split('_')[0],
                        'dataset': dataset_name,
                        'side': meta['side'].iloc[i] if 'side' in meta.columns else 'unknown',
                        'electrode': electrode,
                        'depth': meta['depth'].iloc[i] if 'depth' in meta.columns else 'unknown',
                        'class': class_val,
                        'length': len(resampled)
                    })
                    imported_count += 1
    
    print(f"\nImport summary:")
    print(f"  Imported: {imported_count} recordings")
    print(f"  Skipped (electrode not allowed): {skipped_count} recordings")
                  
    if all_data and all_meta:
        X = all_data.copy()
        meta_df = pd.concat([pd.DataFrame([m]) for m in all_meta], ignore_index=True)
        return X, meta_df
    else:
        print("Warning: No data imported")
        return None, None


def compute_overlap(meta_df, w=10000, eta_m=0.25, eta_M=0.75, eta_step=0.05, verbose=True):
    """
    Calculate optimal overlap (eta) for window extraction to balance dataset.
    
    This algorithm ensures:
    1. Patient-wise class balancing (equal windows per class for each patient)
    2. Inter-dataset balancing (equal total windows across datasets)
    
    The overlap eta determines how much consecutive windows overlap:
    - eta = 0: no overlap
    - eta = 0.5: 50% overlap
    - eta = 0.75: 75% overlap
    
    Parameters:
    -----------
    meta_df : pd.DataFrame
        Metadata per recording (patient, dataset, side, depth, class, length)
        Includes columns: patient, dataset, side, electrode, depth, class, length, eta, n
    w : int
        Window length in samples (default: 10000 = 1 second at 10kHz)
    eta_m : float
        Minimum overlap (default: 0.25)
    eta_M : float
        Maximum overlap (default: 0.75)
    eta_step : float
        Step size for overlap adjustment (default: 0.05)
    verbose : bool
        Print progress information
    
    Returns:
    --------
    meta_df_e : pd.DataFrame
        Input metadata augmented with 'eta' (overlap) and 'n' (number of windows)
    """
    
    def compute_N(l, w, e):
        """Calculate number of windows given length, window size, and overlap."""
        return np.floor((l - w) / np.round(w * (1 - e))).astype(int) + 1

    np.random.seed(42)

    # Initialize eta values & extract number of windows
    meta_df_e = meta_df.copy()
    meta_df_e['eta'] = eta_m
    meta_df_e['n'] = compute_N(meta_df_e['length'].values, w, eta_m)

    # Step 1: Patient-wise class balancing
    # ------------------------------------
    # For each patient (dataset  d, patient i), adjust overlap to get equal windows for both classes
    d_i = meta_df_e[['dataset', 'patient']].drop_duplicates().reset_index(drop=True)

    for idx, row in d_i.iterrows():
        print(f"\r Step 1 - Balancing patient {idx+1}/{len(d_i)}...", end="", flush=True)
        d, i = row['dataset'], row['patient']
        idx = meta_df_e[(meta_df_e['dataset'] == d) & (meta_df_e['patient'] == i)].index
        subset = meta_df_e.loc[idx]
        
        # Total signal length (l) per class for this patient
        l_di = [
            subset[subset['class'] == 0]['length'].sum(), 
            subset[subset['class'] == 1]['length'].sum()
        ]
        
        # Current number (n) of windows per class
        n_di = [
            subset[subset['class'] == 0]['n'].sum(), 
            subset[subset['class'] == 1]['n'].sum()
        ]
        
        # Find underrepresented class
        c_min = np.argmin(n_di)
        
        # Compute target eta to balance classes
        eta_target = 1 - (l_di[c_min] - w) / (w * n_di[1 - c_min])
        eta_target = np.round(eta_target / eta_step) * eta_step
        
        flag = False
        if eta_target > eta_M:
            eta_target = eta_M
            flag = True
            
        subset.loc[subset['class'] == c_min, 'eta'] = eta_target
        subset.loc[subset['class'] == c_min, 'n'] = compute_N(
            subset[subset['class'] == c_min]['length'].values, w, eta_target
        )
        
        # If imbalance persists, adjust majority class too
        if flag:
            eta_target = 1 - (l_di[1 - c_min] - w) / (w * n_di[c_min])
            eta_target = np.round(eta_target / eta_step) * eta_step
            if eta_target < 0:
                eta_target = 0
            subset.loc[subset['class'] == 1 - c_min, 'eta'] = eta_target
            subset.loc[subset['class'] == 1 - c_min, 'n'] = compute_N(
                subset[subset['class'] == 1 - c_min]['length'].values, w, eta_target
            )

        meta_df_e.loc[subset.index, ['eta', 'n']] = subset[['eta', 'n']]

    # Step 2: Inter-dataset balancing
    # ------------------------------------
    # Adjust overlaps to equalize total windows across datasets
    datasets = meta_df_e['dataset'].unique()
    Nd = [meta_df_e[meta_df_e['dataset'] == d]['n'].sum() for d in datasets]
    N_target = int(np.mean(Nd))

    for d, N_d in zip(datasets, Nd):
        delta_N = N_target - N_d
        patients = meta_df_e[meta_df_e['dataset'] == d]['patient'].unique()
        
        # Distribute window adjustment across patients proportionally
        n_i = np.array([
            meta_df_e[(meta_df_e['dataset'] == d) & (meta_df_e['patient'] == p)]['n'].sum() 
            for p in patients
        ])
        delta_n_i = (n_i / n_i.sum() * delta_N).astype(int)
        n_i_target = n_i + delta_n_i
        
        p_count = 0
        for p, n_t in zip(patients, n_i_target):
            print(f"\r Step 2 - Balancing dataset {d}, patient {p_count+1}/{len(patients)}...", end="", flush=True)
            idx = meta_df_e[(meta_df_e['dataset'] == d) & (meta_df_e['patient'] == p)].index
            subset = meta_df_e.loc[idx]
            l_i = subset['length'].values
            e_i = subset['eta'].values
            n_i = subset['n'].sum()
            
            # Iteratively adjust eta to reach target window count
            while np.abs(n_i - n_t) > 10: # while the difference is significant
                e_i = e_i + np.sign(delta_N) * eta_step
                if (e_i <= eta_M).all() and (e_i >= 0).all():
                    n_i = compute_N(l_i, w, e_i)
                    n_i = n_i.sum()
                else:
                    e_i = np.clip(e_i, 0, eta_M)
                    n_i = compute_N(l_i, w, e_i)
                    n_i = n_i.sum()
                    break
                
            subset['eta'] = e_i
            subset['n'] = compute_N(subset['length'].values, w, e_i)
            meta_df_e.loc[subset.index, ['eta', 'n']] = subset[['eta', 'n']]

            p_count += 1
   
    if verbose:
        print("\nOverlap computation completed:")
        print(f"  Total recordings: {len(meta_df_e)}")
        print(f"  Total windows: {meta_df_e['n'].sum()}")
        for dataset in datasets:
            n_windows = meta_df_e[meta_df_e['dataset'] == dataset]['n'].sum()
            print(f"  {dataset}: {n_windows} windows")
    
    return meta_df_e

def extract_segments(X, meta, w=10000, main_output_path=None):
    """
    Extract fixed-length segments from recordings using computed overlaps.
    
    Parameters:
    -----------
    X : list of np.ndarray
        List of preprocessed signals
    meta : pd.DataFrame
        Metadata with 'eta' (overlap) values
    w : int
        Window length in samples (default: 10000 = 1 second at 10kHz)
    main_output_path : str, optional
        If provided, saves segments to disk organized by dataset/patient
    
    Returns:
    --------
    segments : list of np.ndarray
        Extracted segments
    seg_meta : pd.DataFrame
        Columns: patient, dataset, side, electrode, depth, class, length, eta, n, segment_idx, start, end
        Metadata for each segment (includes original recording info + segment info)
    """
    segments = []
    seg_meta = []
    n_rows = len(meta)
    
    for i, meta_row in meta.iterrows():
        print(f"\rProcessing row {i+1}/{n_rows}...", end="", flush=True)
        signal_data = X[i]
        
        if len(signal_data) < w:
            continue
            
        eta = np.round(meta_row['eta'], 2)
        step = int(np.round(w * (1 - eta)))
        n_windows = int(np.floor((len(signal_data) - w) / step)) + 1
        
        for n in range(n_windows):
            start = n * step
            end = start + w
            if end > len(signal_data):
                end = len(signal_data)
                start = end - w
                
            seg = signal_data[start:end]
            segments.append(seg)
            
            # Create metadata meta_row for this segment
            seg_row = meta_row.copy()
            seg_row['segment_idx'] = n
            seg_row['orig_idx'] = i
            seg_row['start'] = start
            seg_row['end'] = end
            seg_row['eta'] = eta
            seg_meta.append(seg_row)
            
    print("\nFinished segment extraction.")

    seg_meta_df = pd.DataFrame(seg_meta, index=range(len(segments)))

    # Optionally save to disk
    if main_output_path is not None:
        os.makedirs(main_output_path, exist_ok=True)
        
        for dataset in seg_meta_df['dataset'].unique():
            dataset_dir = os.path.join(main_output_path, dataset)
            os.makedirs(dataset_dir, exist_ok=True)
            df_dataset = seg_meta_df[seg_meta_df['dataset'] == dataset]
            patients = df_dataset['patient'].unique()
            n_patients = len(patients)
            
            for p_idx, patient in enumerate(patients, 1):
                print(f"\r  Saving {dataset}: patient {p_idx}/{n_patients}", end="", flush=True)
                
                df_patient = df_dataset[df_dataset['patient'] == patient]
                patient_segments = [segments[idx] for idx in df_patient.index]
                
                # Save all segments for this patient as one array
                segments_array = np.array(patient_segments)
                np.save(os.path.join(dataset_dir, f"{patient}.npy"), segments_array, allow_pickle=True)
                
                # Save metadata
                df_patient.to_csv(os.path.join(dataset_dir, f"{patient}.csv"), index=False)
                
            print(f"\n  Finished saving segments for dataset {dataset}.")

    return segments, seg_meta_df


def import_signals_and_meta_from_folder(main_path, dataset_name_list=None):
    """
    Import previously extracted and saved segments from folder structure.
    
    Useful for loading preprocessed data without recomputing everything.
    
    Parameters:
    -----------
    main_path : str
        Root directory containing dataset folders
    dataset_name_list : list of str, optional
        Datasets to load (default: ['AlphaMap', 'NeuroSet', 'NeuroOmega'])
    
    Returns:
    --------
    signals : np.ndarray
        Stacked array of all segments
    meta : pd.DataFrame
        Concatenated metadata for all segments
    """
    signals_list = []
    meta_tables = []

    if dataset_name_list is None:
        dataset_name_list = ['AlphaMap', 'NeuroSet', 'NeuroOmega']

    for dataset_name in dataset_name_list:
        dataset_folder = os.path.join(main_path, dataset_name)
        
        if not os.path.exists(dataset_folder):
            print(f"Warning: Dataset folder not found: {dataset_folder}")
            continue
            
        npy_files = [fname for fname in os.listdir(dataset_folder) if fname.endswith('.npy')]
        total = len(npy_files)
        
        for idx, fname in enumerate(npy_files, 1):
            patient_id = fname.replace('.npy', '')
            signal_data = np.load(os.path.join(dataset_folder, fname), allow_pickle=True)
            meta_path = os.path.join(dataset_folder, f"{patient_id}.csv")
            
            if os.path.exists(meta_path):
                meta_df = pd.read_csv(meta_path)
                meta_tables.append(meta_df)
                signals_list.append(signal_data)
            else:
                print(f"\nWarning: Metadata not found for {patient_id}")
                
            print(f"\rLoading {dataset_name}: {idx}/{total} patients", end='', flush=True)
        print()
        
    signals = np.concatenate(signals_list, axis=0)
    meta = pd.concat(meta_tables, ignore_index=True)
    return signals, meta


def split_patients_into_balanced_folds(patients_list, n_folds):
    """
    Split patients into n_folds with balanced segment counts using greedy algorithm.
    
    Parameters:
    -----------
    patients_list : list of dict
        Each dict contains {'patient', 'dataset', 'n_segs'}
    n_folds : int
        Number of folds to create
    
    Returns:
    --------
    folds : list of list
        Each fold is a list of patient dicts
    """
    if len(patients_list) == 0:
        return [[] for _ in range(n_folds)]
    
    # Sort by segment count (descending) for better greedy assignment
    patients_sorted = sorted(patients_list, key=lambda x: x['n_segs'], reverse=True)
    
    # Initialize folds
    folds = [[] for _ in range(n_folds)]
    fold_totals = [0] * n_folds
    
    # Greedy assignment: assign each patient to fold with minimum segments
    for patient_info in patients_sorted:
        min_fold_idx = np.argmin(fold_totals)
        folds[min_fold_idx].append(patient_info)
        fold_totals[min_fold_idx] += patient_info['n_segs']
    
    return folds


def create_construction_set(all_patients, test_set_patients):
    """
    Create construction set by removing test patients from all patients.
    
    Parameters:
    -----------
    all_patients : pd.DataFrame
        All available patients
    test_set_patients : list of dict [{patient:..., dataset:...}, ...]
        Test set patient information
    
    Returns:
    --------
    construction_patients : list of dict
        Remaining patients for training/validation
    """
    test_patient_pairs = set()
    for patient_info in test_set_patients:
        test_patient_pairs.add((patient_info['patient'], patient_info['dataset']))
    
    construction_patients = []
    for _, row in all_patients.iterrows():
        patient_pair = (row['patient'], row['dataset'])
        if patient_pair not in test_patient_pairs:
            construction_patients.append({
                'patient': row['patient'],
                'dataset': row['dataset'],
                'n_segs': row['n_segs']
            })
    
    return construction_patients


def split_construction_into_folds(construction_patients, n_folds, option_num, meta, main_output_path=None):
    """
    Split construction set into n_folds for cross-validation.
    
    Parameters:
    -----------
    construction_patients : list of dict
        Patients in construction set
    n_folds : int
        Number of folds (5 or 10)
    option_num : int
        Test set option number (for file naming)
    meta : pd.DataFrame
        Full metadata (for window counting)
    
    Returns:
    --------
    folds : dict
        Dictionary with fold assignments and statistics
    """
    # Group by dataset
    datasets = {}
    for patient_info in construction_patients:
        dataset = patient_info['dataset']
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(patient_info)
    
    # Split each dataset independently
    dataset_folds = {}
    for dataset, patients_list in datasets.items():
        dataset_folds[dataset] = split_patients_into_balanced_folds(patients_list, n_folds)
    
    # Combine folds across datasets
    folds = {}
    for fold_idx in range(n_folds):
        folds[f'fold_{fold_idx+1}'] = []
        fold_segment_counts = {}
        fold_window_counts = {}
        
        for dataset in dataset_folds:
            if fold_idx < len(dataset_folds[dataset]):
                for patient_info in dataset_folds[dataset][fold_idx]:
                    folds[f'fold_{fold_idx+1}'].append((
                        patient_info['patient'],
                        patient_info['dataset'],
                        patient_info['n_segs']
                    ))
                    
                    # Count segments and windows
                    dataset_name = patient_info['dataset']
                    fold_segment_counts[dataset_name] = fold_segment_counts.get(dataset_name, 0) + patient_info['n_segs']
                    
                    idx = meta[(meta['patient'] == patient_info['patient']) & 
                             (meta['dataset'] == dataset_name)].index
                    n_windows = len(idx)
                    fold_window_counts[dataset_name] = fold_window_counts.get(dataset_name, 0) + n_windows
        
        folds[f'fold_{fold_idx+1}_segments'] = fold_segment_counts
        folds[f'fold_{fold_idx+1}_windows'] = fold_window_counts
    
    # Save to CSV
    fold_rows = []
    for fold_name, patients in folds.items():
        if not fold_name.endswith('_segments') and not fold_name.endswith('_windows'):
            for patient, dataset, n_segs in patients:
                fold_rows.append({
                    'fold': fold_name,
                    'patient': patient,
                    'dataset': dataset,
                    'n_segs': n_segs
                })
    
    if fold_rows:
        fold_df = pd.DataFrame(fold_rows)
        filename = f"construction_folds_{n_folds}_option_{option_num}.csv"
        if main_output_path is not None:
            filename = os.path.join(main_output_path, filename)
        fold_df.to_csv(filename, index=False)
        
        # Print balance statistics
        print(f"\nConstruction set {n_folds}-fold split (Option {option_num}):")
        all_segment_counts = []
        for fold_idx in range(n_folds):
            fold_name = f'fold_{fold_idx+1}'
            if f'{fold_name}_segments' in folds:
                total_segs = sum(folds[f'{fold_name}_segments'].values())
                all_segment_counts.append(total_segs)
                print(f"  {fold_name}: {total_segs} segments across {len(folds[fold_name])} patients")
        
        if all_segment_counts:
            balance_cv = (np.std(all_segment_counts) / np.mean(all_segment_counts)) * 100
            print(f"  Balance CV: {balance_cv:.2f}%")
    
    return folds


def generate_random_balanced_test_set(patients, test_size, target_segments_per_dataset, seed):
    """
    Generate a single random balanced test set.
    
    Uses random sampling with multiple trials to find the most balanced combination.
    
    Parameters:
    -----------
    patients : pd.DataFrame
        All available patients with segment counts
    test_size : int
        Target number of patients per dataset
    target_segments_per_dataset : float
        Target number of segments per dataset
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    test_patients : list of dict
        Selected test patients
    """
    np.random.seed(seed)
    test_patients = []
    
    for dataset in patients['dataset'].unique():
        df = patients[patients['dataset'] == dataset].copy()
        
        # Try multiple random combinations
        best_combination = None
        best_distance = float('inf')
        n_trials = min(1000, len(df) * 10)
        
        for _ in range(n_trials):
            if len(df) <= test_size:
                sample_indices = df.index.tolist()
            else:
                sample_indices = np.random.choice(df.index, size=test_size, replace=False)
            
            total_segments = df.loc[sample_indices, 'n_segs'].sum()
            distance = abs(total_segments - target_segments_per_dataset)
            
            if distance < best_distance:
                best_distance = distance
                best_combination = sample_indices
        
        # Add best combination
        for idx in best_combination:
            test_patients.append({
                'patient': df.loc[idx, 'patient'],
                'dataset': dataset,
                'n_segs': df.loc[idx, 'n_segs']
            })
    
    return test_patients


def create_assignment_summary_table(all_patients, all_test_sets, main_output_path=None):
    """
    Create comprehensive CSV showing all patient assignments across all options.
    
    Parameters:
    -----------
    all_patients : pd.DataFrame
        All available patients
    all_test_sets : list of dict
        All test set options with fold information
    
    Returns:
    --------
    summary_df : pd.DataFrame
        Complete assignment table
    """
    summary_rows = []
    
    for _, row in all_patients.iterrows():
        patient_dataset = f"{row['patient']}_{row['dataset']}"
        summary_row = {
            'patient_dataset': patient_dataset,
            'patient': row['patient'],
            'dataset': row['dataset'],
            'n_segs': row['n_segs']
        }
        
        # Initialize columns
        for i in range(len(all_test_sets)):
            summary_row[f'test_{i}'] = 0
            summary_row[f'test_{i}_5fold'] = -1
            summary_row[f'test_{i}_10fold'] = -1
        
        summary_rows.append(summary_row)
    
    # Fill in assignments
    for test_idx, test_set_info in enumerate(all_test_sets):
        test_patients_set = set()
        for patient_info in test_set_info['test_patients']:
            patient_dataset = f"{patient_info['patient']}_{patient_info['dataset']}"
            test_patients_set.add(patient_dataset)
            
            for row in summary_rows:
                if row['patient_dataset'] == patient_dataset:
                    row[f'test_{test_idx}'] = 1
                    break
        
        # Fill 5-fold assignments
        folds_5 = test_set_info['folds_5']
        for fold_name, patients_list in folds_5.items():
            if not fold_name.endswith('_segments') and not fold_name.endswith('_windows'):
                fold_num = int(fold_name.split('_')[1])
                for patient, dataset, _ in patients_list:
                    patient_dataset = f"{patient}_{dataset}"
                    for row in summary_rows:
                        if row['patient_dataset'] == patient_dataset:
                            row[f'test_{test_idx}_5fold'] = fold_num
                            break
        
        # Fill 10-fold assignments
        folds_10 = test_set_info['folds_10']
        for fold_name, patients_list in folds_10.items():
            if not fold_name.endswith('_segments') and not fold_name.endswith('_windows'):
                fold_num = int(fold_name.split('_')[1])
                for patient, dataset, _ in patients_list:
                    patient_dataset = f"{patient}_{dataset}"
                    for row in summary_rows:
                        if row['patient_dataset'] == patient_dataset:
                            row[f'test_{test_idx}_10fold'] = fold_num
                            break
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(['dataset', 'patient']).reset_index(drop=True)
    
    filename = "patient_assignments_summary.csv"
    if main_output_path is not None:
        filename = os.path.join(main_output_path, filename)
    summary_df.to_csv(filename, index=False)
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE ASSIGNMENT TABLE SAVED: {filename}")
    print(f"Table contains {len(summary_df)} patient-dataset combinations")
    
    return summary_df


def split_constr_test(segments, meta, main_output_path=None, test_size=10):
    """
    Generate multiple random test/construction splits with cross-validation folds.
    
    Creates 5 different random splits (different seeds) to allow selection of best option.
    Each split includes:
    - Balanced test set
    - Construction set (remaining patients)
    - 5-fold CV splits
    - 10-fold CV splits
    
    Parameters:
    -----------
    segments : list/array
        Extracted segments (not used in splitting logic)
    meta : pd.DataFrame
        Segment metadata with patient, dataset info
    test_size : int
        Target number of test patients per dataset
    
    Returns:
    --------
    folds : dict
        Complete split information for all 5 optionselectrode', '
    """

    if main_output_path is not None:
        os.makedirs(main_output_path, exist_ok=True)

    recordings = meta[['patient', 'dataset', 'side', 'depth', 'n']].drop_duplicates().reset_index(drop=True)
    patients = recordings[['patient', 'dataset']].drop_duplicates().reset_index(drop=True)
    
    # Compute segments per patient
    for patient, dataset in patients.itertuples(index=False):
        idx = recordings[(recordings['patient'] == patient) & (recordings['dataset'] == dataset)].index
        n_segs = recordings.loc[idx, 'n'].sum() if 'n' in recordings.columns else len(idx)
        patients.loc[(patients['patient'] == patient) & (patients['dataset'] == dataset), 'n_segs'] = n_segs
    
    # Calculate targets
    datasets = patients['dataset'].unique()
    total_segments_available = {}
    for dataset in datasets:
        df = patients[patients['dataset'] == dataset]
        total_segments_available[dataset] = df['n_segs'].sum()
    
    target_segments_per_dataset = np.mean([v * 0.2 for v in total_segments_available.values()])
    
    print(f"Target segments per dataset: {target_segments_per_dataset:.1f}")
    print(f"Available segments per dataset: {total_segments_available}")
    
    # Generate 5 options with different seeds
    seeds = [42, 123, 456, 789, 999]
    all_test_sets = []
    
    for i, seed in enumerate(seeds):
        test_set = generate_random_balanced_test_set(patients, test_size, target_segments_per_dataset, seed)
        
        # Calculate balance metrics
        segment_counts = {}
        window_counts = {}
        for patient_info in test_set:
            dataset = patient_info['dataset']
            patient = patient_info['patient']
            
            segment_counts[dataset] = segment_counts.get(dataset, 0) + patient_info['n_segs']
            
            idx = meta[(meta['patient'] == patient) & (meta['dataset'] == dataset)].index
            n_windows = len(idx)
            window_counts[dataset] = window_counts.get(dataset, 0) + n_windows
        
        segment_values = list(segment_counts.values())
        balance_variance = np.var(segment_values)
        balance_std = np.std(segment_values)
        balance_cv = (balance_std / np.mean(segment_values)) * 100
        
        all_test_sets.append({
            'option': i + 1,
            'seed': seed,
            'test_patients': test_set,
            'segment_counts': segment_counts,
            'window_counts': window_counts,
            'balance_variance': balance_variance,
            'balance_std': balance_std,
            'balance_cv': balance_cv
        })
        
        # Save test set
        test_set_rows = []
        for patient_info in test_set:
            test_set_rows.append({
                'patient': patient_info['patient'],
                'dataset': patient_info['dataset'], 
                'n_segs': patient_info['n_segs']
            })
        
        test_set_df = pd.DataFrame(test_set_rows)
        filename = f"test_set_option_{i+1}_seed_{seed}.csv"
        if main_output_path is not None:
            filename = os.path.join(main_output_path, filename)
        test_set_df.to_csv(filename, index=False)
        print(f"\nSaved: {filename}")
        
        # Create construction set
        construction_patients = create_construction_set(patients, test_set)
        
        # Generate CV folds
        folds_5 = split_construction_into_folds(construction_patients, n_folds=5, option_num=i+1, meta=meta, main_output_path=main_output_path)
        folds_10 = split_construction_into_folds(construction_patients, n_folds=10, option_num=i+1, meta=meta, main_output_path=main_output_path)
        
        all_test_sets[i]['construction_patients'] = construction_patients
        all_test_sets[i]['folds_5'] = folds_5
        all_test_sets[i]['folds_10'] = folds_10
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL TEST SET OPTIONS:")
    for test_set_info in all_test_sets:
        print(f"\nOption {test_set_info['option']} (seed={test_set_info['seed']}):")
        print(f"  Test set balance CV: {test_set_info['balance_cv']:.1f}%")
        print(f"  Test segments per dataset: {test_set_info['segment_counts']}")
        print(f"  Construction set: {len(test_set_info['construction_patients'])} patients")
    
    # Create comprehensive table
    create_assignment_summary_table(patients, all_test_sets, main_output_path=main_output_path)
    
    # Return in original format
    folds = {}
    for i in range(len(all_test_sets)):
        folds[f'test_{i+1}'] = [(d['patient'], d['dataset'], d['n_segs']) for d in all_test_sets[i]['test_patients']]
        folds[f'construction_{i+1}'] = [(d['patient'], d['dataset'], d['n_segs']) for d in all_test_sets[i]['construction_patients']]
        folds[f'folds_5_{i+1}'] = all_test_sets[i]['folds_5']
        folds[f'folds_10_{i+1}'] = all_test_sets[i]['folds_10']
    
    return folds


def split_constr_test_fixed(segments, meta, main_output_path=None, test_size=10, fixed_test_pairs=None):
    """
    Like split_constr_test, but with specific patients forced into test set.
    
    Useful when you want to ensure certain patients are always in test set
    (e.g., edge cases, specific recording conditions).
    
    Parameters:
    -----------
    segments : list/array
        Extracted segments
    meta : pd.DataFrame
        Segment metadata
    test_size : int
        Additional patients to sample per dataset (beyond fixed ones)
    fixed_test_pairs : list of tuple
        (patient, dataset) pairs to force into test set
    
    Returns:
    --------
    folds : dict
        Complete split information for all 5 options
    """
    if fixed_test_pairs is None:
        fixed_test_pairs = []

    if main_output_path is not None:
        os.makedirs(main_output_path, exist_ok=True)

    recordings = meta[['patient', 'dataset', 'side', 'depth','electrode', 'n']].drop_duplicates().reset_index(drop=True)
    patients = recordings[['patient', 'dataset']].drop_duplicates().reset_index(drop=True)

    # Compute segments per patient
    for patient, dataset in patients.itertuples(index=False):
        idx = recordings[(recordings['patient'] == patient) & (recordings['dataset'] == dataset)].index
        n_segs = recordings.loc[idx, 'n'].sum() if 'n' in recordings.columns else len(idx)
        patients.loc[(patients['patient'] == patient) & (patients['dataset'] == dataset), 'n_segs'] = n_segs

    # Calculate targets
    datasets = patients['dataset'].unique()
    total_segments_available = {}
    for dataset in datasets:
        df = patients[patients['dataset'] == dataset]
        total_segments_available[dataset] = df['n_segs'].sum()

    target_segments_per_dataset = np.mean([v * 0.2 for v in total_segments_available.values()])

    print(f"Target segments per dataset: {target_segments_per_dataset:.1f}")
    print(f"Available segments per dataset: {total_segments_available}")

    # Organize fixed patients by dataset
    fixed_by_dataset = defaultdict(list)
    for p, d in fixed_test_pairs:
        fixed_by_dataset[d].append(p)

    # Generate 5 options
    seeds = [42, 123, 456, 789, 999]
    all_test_sets = []

    for i, seed in enumerate(seeds):
        np.random.seed(seed)
        test_set = []

        for dataset in datasets:
            df = patients[patients['dataset'] == dataset].copy()
            fixed_patients_in_d = fixed_by_dataset.get(dataset, [])

            # Add fixed patients first
            for fp in fixed_patients_in_d:
                match = df[df['patient'] == fp]
                if len(match) > 0:
                    test_set.append({
                        'patient': fp,
                        'dataset': dataset,
                        'n_segs': match.iloc[0]['n_segs']
                    })

            # Sample additional patients
            remaining_df = df[~df['patient'].isin(fixed_patients_in_d)].copy()
            additional_needed = max(0, min(test_size, len(df)) - len(fixed_patients_in_d))

            if additional_needed > 0 and len(remaining_df) > 0:
                additional_needed = min(additional_needed, len(remaining_df))
                sampled = remaining_df.sample(n=additional_needed, random_state=seed)
                for _, row in sampled.iterrows():
                    test_set.append({
                        'patient': row['patient'],
                        'dataset': dataset,
                        'n_segs': row['n_segs']
                    })

        # Calculate balance metrics
        segment_counts = {}
        window_counts = {}
        for patient_info in test_set:
            dataset = patient_info['dataset']
            patient = patient_info['patient']
            segment_counts[dataset] = segment_counts.get(dataset, 0) + patient_info['n_segs']
            idx = meta[(meta['patient'] == patient) & (meta['dataset'] == dataset)].index
            n_windows = len(idx)
            window_counts[dataset] = window_counts.get(dataset, 0) + n_windows

        segment_values = list(segment_counts.values()) if segment_counts else [0]
        balance_variance = np.var(segment_values)
        balance_std = np.std(segment_values)
        balance_cv = (balance_std / np.mean(segment_values)) * 100 if np.mean(segment_values) > 0 else 0.0

        all_test_sets.append({
            'option': i + 1,
            'seed': seed,
            'test_patients': test_set,
            'segment_counts': segment_counts,
            'window_counts': window_counts,
            'balance_variance': balance_variance,
            'balance_std': balance_std,
            'balance_cv': balance_cv
        })

        # Save test set
        test_set_rows = []
        for patient_info in test_set:
            test_set_rows.append({
                'patient': patient_info['patient'],
                'dataset': patient_info['dataset'],
                'n_segs': patient_info['n_segs']
            })

        test_set_df = pd.DataFrame(test_set_rows)
        filename = f"test_set_option_{i+1}_seed_{seed}.csv"
        if main_output_path is not None:
            filename = os.path.join(main_output_path, filename)
        test_set_df.to_csv(filename, index=False)
        print(f"\nSaved: {filename}")

        # Create construction set and folds
        construction_patients = create_construction_set(patients, test_set)
        folds_5 = split_construction_into_folds(construction_patients, n_folds=5, option_num=i+1, meta=meta, main_output_path=main_output_path)
        folds_10 = split_construction_into_folds(construction_patients, n_folds=10, option_num=i+1, meta=meta, main_output_path=main_output_path)

        all_test_sets[i]['construction_patients'] = construction_patients
        all_test_sets[i]['folds_5'] = folds_5
        all_test_sets[i]['folds_10'] = folds_10

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL FIXED TEST SET OPTIONS:")
    for test_set_info in all_test_sets:
        print(f"\nOption {test_set_info['option']} (seed={test_set_info['seed']}):")
        print(f"  Test set balance CV: {test_set_info['balance_cv']:.1f}%")
        print(f"  Test segments per dataset: {test_set_info['segment_counts']}")
        print(f"  Construction set: {len(test_set_info['construction_patients'])} patients")

    # Create comprehensive table
    create_assignment_summary_table(patients, all_test_sets, main_output_path=main_output_path)

    # Return in original format
    folds = {}
    for i in range(len(all_test_sets)):
        folds[f'test_{i+1}'] = [(d['patient'], d['dataset'], d['n_segs']) for d in all_test_sets[i]['test_patients']]
        folds[f'construction_{i+1}'] = [(d['patient'], d['dataset'], d['n_segs']) for d in all_test_sets[i]['construction_patients']]
        folds[f'folds_5_{i+1}'] = all_test_sets[i]['folds_5']
        folds[f'folds_10_{i+1}'] = all_test_sets[i]['folds_10']

    return folds


def main():
    """
    Main execution function.
    
    Modify the parameters below to suit your dataset and requirements.
    """
    # Configuration
    base_path = 'D:/2025_UniTO_MERs_29052025'
    dataset_names = ['NeuroOmega', 'AlphaMap', 'NeuroSet']
    fsamp = {'NeuroOmega': 24000, 'AlphaMap': 12000, 'NeuroSet': 20000}
    out_path = 'dataset_processed_windows'
    meta_overlap_path = "meta_overlap.csv"
    
    # Check if processed data already exists
    if os.path.exists(out_path):
        if any(os.path.isdir(os.path.join(out_path, d)) for d in os.listdir(out_path)):
            print("Loading existing processed data...")
            segments, seg_meta = import_signals_and_meta_from_folder(out_path, dataset_name_list=dataset_names)
        else:
            print("Output path exists but is empty, processing from scratch...")
            segments, seg_meta = None, None
    else:
        segments, seg_meta = None, None
    
    # Process if needed
    if segments is None:
        # Import config to get allowed electrodes
        import config
        
        X, meta = import_dataset(base_path, dataset_names, fsamp, 
                                allowed_electrodes=config.ALLOWED_ELECTRODES)
        print("Processing raw dataset...")
        
        if X is None or meta is None:
            print("Error: Failed to import dataset")
            return
        
        # Compute or load overlap values
        if os.path.isfile(meta_overlap_path):
            print(f"Loading overlap values from {meta_overlap_path}")
            meta = pd.read_csv(meta_overlap_path)
        else:
            print("Computing overlap values...")
            meta = compute_overlap(meta, w=10000, eta_m=0.25, eta_M=0.75, eta_step=0.05, verbose=True)
            meta.to_csv(meta_overlap_path, index=False)
            print(f"Saved overlap values to {meta_overlap_path}")
        
        # Extract segments
        print("Extracting segments...")
        segments, seg_meta = extract_segments(X, meta, w=10000, main_output_path=out_path)
    
    # Generate splits
    print("\nGenerating test/construction splits...")
    
    # Option 1: Random balanced splits
    folds = split_constr_test(segments, seg_meta, main_output_path=out_path, test_size=10)
    
    # Option 2: Fixed test patients (uncomment and modify as needed)
    # fixed_test_pairs = [
    #     ('P12', 'NeuroSet'), ('P011', 'AlphaMap'), ('P04', 'Ciecierski'),
    #     ('P17', 'NeuroSet'), ('P012', 'AlphaMap'), ('P05', 'Ciecierski'),
    #     ('P18', 'NeuroSet'), ('P016', 'AlphaMap'), ('P11', 'Ciecierski'),
    #     ('P20', 'NeuroSet'), ('P026', 'AlphaMap'), ('P14', 'Ciecierski'),
    #     ('P21', 'NeuroSet'), ('P041', 'AlphaMap'), ('P17', 'Ciecierski'),
    #     ('P23', 'NeuroSet'), ('P052', 'AlphaMap'), ('P20', 'Ciecierski'),
    #     ('P27', 'NeuroSet'), ('P069', 'AlphaMap'), ('P25', 'Ciecierski'),
    #     ('P28', 'NeuroSet'), ('P081', 'AlphaMap'), ('P28', 'Ciecierski'),
    #     ('P33', 'NeuroSet'), ('P083', 'AlphaMap'), ('P36', 'Ciecierski'),
    #     ('P35', 'NeuroSet'), ('P085', 'AlphaMap'), ('P41', 'Ciecierski'),
    # ]
    # folds = split_constr_test_fixed(segments, seg_meta, main_output_path=out_path, test_size=10, 
    #                                 fixed_test_pairs=fixed_test_pairs)
    
    print('\nDataset preparation complete!')
    print('Generated files:')
    print('  - test_set_option_*.csv (5 options)')
    print('  - construction_folds_*.csv (5-fold and 10-fold CV)')
    print('  - patient_assignments_summary.csv (comprehensive table)')


if __name__ == '__main__':
    main()
