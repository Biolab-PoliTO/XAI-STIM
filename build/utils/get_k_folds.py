import numpy as np
import pandas as pd
import os

def get_kFolds(windows_final, k=5):
    """
    Splits patients in each dataset into k folds with approximately equal number of windows,
    then plots a stacked bar plot per dataset showing the number of windows per fold.

    Args:
        windows_final (dict): Dict of DataFrames, one per dataset, indexed by patient, columns are classes.
        k (int): Number of folds.
    """

    folds = {}

    for dname, df in windows_final.items():
        df = df.copy() 
        df['Total'] = df.sum(axis=1) # Compute total windows per patient (row-wise sum)
        df_sorted = df.sort_values('Total', ascending=False) # Sort patients by total windows (descending)
        patients = df_sorted.index.tolist()
        totals = df_sorted['Total'].values

        # Greedy assignment to k folds to balance total windows
        folds[dname] = []
        fold_totals = [0] * k
        fold_patients = [[] for _ in range(k)]
        for i, (pt, tw) in enumerate(zip(patients, totals)):
            idx = np.argmin(fold_totals)
            fold_patients[idx].append(pt)
            fold_totals[idx] += tw

        # For each fold, sum windows per class
        fold_class_windows = []
        for pts in fold_patients:
            if pts:
                fold_sum = df.loc[pts, df.columns[:-1]].sum()
            else:
                fold_sum = pd.Series([0]* (df.shape[1]-1), index=df.columns[:-1])
            fold_class_windows.append(fold_sum)

        # Print patient numbers for each fold
        for i, pts in enumerate(fold_patients):
            folds[dname].append(pts)

    return folds

def get_num_windows(all_datasets):
    windows = {}
    for dataset_name, data in all_datasets.items():
        meta = data['meta'].copy()
        patients = meta['patient'].unique()
        classes = meta['class'].unique()
        windows_table = pd.DataFrame(0, index=patients, columns=classes)
        for patient in patients:
            for cls in classes:
                count = meta[(meta['patient'] == patient) & (meta['class'] == cls)].shape[0]
                windows_table.loc[patient, cls] = count
        windows[dataset_name] = windows_table
    return windows

def save_k_fold_assignments(folds, main_path=''):
    # Save fold assignments for each dataset
    if main_path:
        if not os.path.exists(main_path):
            os.makedirs(main_path)
    for dataset_name, patient_to_fold in folds.items():
        # patient_to_fold is a list of lists: each index is the fold, each value is a list of patients
        fold_to_patients = {fold: [str(patient) for patient in patients] for fold, patients in enumerate(patient_to_fold)}
        # Create DataFrame: one row per fold, patients as "_" separated string
        fold_df = pd.DataFrame([
            {'fold': fold, 'patients': '_'.join(sorted(patients))}
            for fold, patients in sorted(fold_to_patients.items())
        ])
        fold_csv_path = os.path.join(main_path, f'{dataset_name}_fold_assignments.csv')
        fold_df.to_csv(fold_csv_path, index=False)
        print(f'Saved fold assignments for {dataset_name} to {fold_csv_path}')

def __main__():

    # dataset_path = '//192.168.164.127/public/projects/2024_UniTO_MERs_Classification/2025_XXX_XAI_STN_MERs/Scripts/cross_validation_datasets'
    dataset_path = 'D:/2025_UniTO_MERs_29052025/cross_validation_datasets'
    
    dataset_label = ['AlphaMap','NeuroSet','Ciecierski']
    all_datasets = {}
    for dataset_name in dataset_label:
        signals = np.load(f'{dataset_path}/{dataset_name}_signals.npy', allow_pickle=True)
        meta = pd.read_csv(f'{dataset_path}/{dataset_name}_meta.csv')
        all_datasets[dataset_name] = {
            'signals': signals,
            'meta': meta
        }
        print(f'Loaded dataset: {dataset_name}')

    # Concatenate all meta DataFrames and add dataset_name column
    meta_tables = []
    for dataset_name, data in all_datasets.items():
        meta_df = data['meta'].copy()
        meta_df['dataset'] = dataset_name
        meta_tables.append(meta_df)
    all_meta = pd.concat(meta_tables, ignore_index=True)

    # All-in-one cross-validation and test sets
    # ---------------------------------------------------------------------------------------
    num_samples_pt_class = get_num_windows(all_datasets)
    folds = get_kFolds(num_samples_pt_class, k=5)

    # for dataset_name, fold_lists in folds.items():
    #     print(f"Dataset: {dataset_name}")
    #     for i, patients in enumerate(fold_lists):
    #         print(f"  Fold {i+1}: {sorted(patients)}")
    #     print()

    for dataset_name in folds.keys():
        patient_to_fold = folds[dataset_name]
        all_meta.loc[all_meta['dataset'] == dataset_name, 'test'] = all_meta['patient'].map({patient: fold for fold, patients in enumerate(patient_to_fold) for patient in patients})

    # Filter out signals with fold "4" for test column
    for k_fold in (3, 5, 10):
        for test in range(5):
            filtered_datasets = {}
            for dataset_name, data in all_datasets.items():
                meta = all_meta[all_meta['dataset'] == dataset_name]
                mask = meta['test'] != test
                filtered_meta = meta[mask].copy()
                filtered_signals = data['signals'][mask.values]
                filtered_datasets[dataset_name] = {
                    'signals': filtered_signals,
                    'meta': filtered_meta
                }

            num_samples_pt_class = get_num_windows(filtered_datasets)
            folds = get_kFolds(num_samples_pt_class, k=k_fold)

            # Add test_5_5_fold column to all_meta
            all_meta[f'test_{test+1}_{k_fold}_fold'] = -1
            for dataset_name in folds.keys():
                patient_to_fold = folds[dataset_name]
                # Only assign fold numbers to patients present in filtered_meta (i.e., not class "test")
                filtered_patients = set(filtered_datasets[dataset_name]['meta']['patient'].unique())
                patient_fold_map = {patient: fold for fold, patients in enumerate(patient_to_fold) for patient in patients if patient in filtered_patients}
                mask = (all_meta['dataset'] == dataset_name) & (all_meta['patient'].isin(filtered_patients))
                all_meta.loc[mask, f'test_{test+1}_{k_fold}_fold'] = all_meta.loc[mask, 'patient'].map(patient_fold_map)

            print(f'Finished processing test_{test+1}_{k_fold}_fold')

    # leave-one-out/one-vs-all fold assignments
    # ---------------------------------------------------------------------------------------
    dataset = {}
    for dataset_name in all_datasets.keys():      
        dataset = {dataset_name: all_datasets[dataset_name]}
        num_samples_pt_class = get_num_windows(dataset)
        for k_folds in (5, 10):
            folds = get_kFolds(num_samples_pt_class, k=k_folds)
            for dataset_name in folds.keys():
                patient_to_fold = folds[dataset_name]
                all_meta.loc[all_meta['dataset'] == dataset_name, f'{k_folds}_fold_cv'] = all_meta['patient'].map({patient: fold for fold, patients in enumerate(patient_to_fold) for patient in patients})

    all_meta.to_excel('all_folds.xlsx', index=False)
        
    print('Done')

if __name__ == "__main__":
    __main__()
