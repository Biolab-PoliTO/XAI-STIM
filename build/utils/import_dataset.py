import numpy as np
import pandas as pd
import os
import sys


def import_data(dataset_path=None):

    print('Loading Dataset...')
    if dataset_path is None:
        # dataset_path = '//192.168.164.127/public/projects/2024_UniTO_MERs_Classification/2025_XXX_XAI_STN_MERs/Work/cross_validation_datasets'
        dataset_path = 'D:/2025_UniTO_MERs_29052025/cross_validation_datasets'
    
    dataset_label = ['AlphaMap','NeuroSet','Ciecierski']
    signals_list = []
    meta_tables = []
    for dataset_name in dataset_label:
        signals = np.load(f'{dataset_path}/{dataset_name}_signals.npy', allow_pickle=True)
        meta = pd.read_csv(f'{dataset_path}/{dataset_name}_meta.csv')
        meta['dataset'] = dataset_name  # Add dataset name column
        signals_list.append(signals)
        meta_tables.append(meta)
        del signals, meta
        print(f'Loaded {dataset_name}')
    print()
    all_signals = np.concatenate(signals_list, axis=0)
    all_meta = pd.concat(meta_tables, ignore_index=True)
    del signals_list, meta_tables

    return all_signals, all_meta

def import_signals_and_meta_from_folder(main_path, dataset_name_list=None):
    """
    Imports and stacks all patient signals (.npy) and meta (.csv) files from a dataset folder.

    Args:
        main_path (str): The root directory containing the dataset folders.
        dataset_name (str): The name of the dataset folder (e.g., 'AlphaMap').

    Returns:
        signals (np.ndarray): Stacked array of all signals.
        meta (pd.DataFrame): Concatenated DataFrame of all meta tables.
    """
    
    signals_list = []
    meta_tables = []

    if dataset_name_list is None:
        dataset_name_list = ['AlphaMap','NeuroSet','Ciecierski']

    for dataset_name in dataset_name_list:
        dataset_folder = os.path.join(main_path, dataset_name)
        npy_files = [fname for fname in os.listdir(dataset_folder) if fname.endswith('.npy')]
        total = len(npy_files)
        for idx, fname in enumerate(npy_files, 1):
            patient_id = fname.replace('.npy', '')
            signal = np.load(os.path.join(dataset_folder, fname), allow_pickle=True)
            meta_path = os.path.join(dataset_folder, f"{patient_id}.csv")
            if os.path.exists(meta_path):
                meta = pd.read_csv(meta_path)
                meta['patient'] = patient_id
                meta['dataset'] = dataset_name
                signals_list.append(signal)
                meta_tables.append(meta)
            print(f"\rLoading {dataset_name}: {idx}/{total} patients", end='', flush=True)
        print()  # for newline after progress
    signals = np.concatenate(signals_list, axis=0)
    meta = pd.concat(meta_tables, ignore_index=True)
    return signals, meta

if __name__ == "__main__":

    signals, meta = import_data()

    for dataset_name in meta['dataset'].unique():
        dataset_folder = os.path.join(dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)
        dataset_mask = meta['dataset'] == dataset_name
        dataset_meta = meta[dataset_mask]
        dataset_signals = signals[dataset_mask.values]
        for patient in dataset_meta['patient'].unique():
            patient_mask = dataset_meta['patient'] == patient
            patient_signals = dataset_signals[patient_mask.values]
            patient_meta = dataset_meta[patient_mask]
            np.save(os.path.join(dataset_folder, f"{patient}.npy"), patient_signals)
            patient_meta.to_csv(os.path.join(dataset_folder, f"{patient}.csv"), index=False)
            print(f"Saved {dataset_name}/{patient}.npy and .csv")

    print(f"Signals shape: {signals.shape}")
    print(f"Meta shape: {meta.shape}")