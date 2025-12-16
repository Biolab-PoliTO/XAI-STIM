import numpy as np
import pandas as pd
import os
import torch
import utils
import utils.networks as networks
from datetime import datetime

if __name__ == "__main__":

    # A. INITIALIZATIONS
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    fsamp = 10000
    processing_opts = {'fsamp': fsamp, 
                       'normalization': 'zscore',
                       'filter_type': ['highpass','notch'],
                       'notch_freqs': np.arange(250, 3000, 50), #np.arange(100, 4000, 50),
                       'lowcut': 200, #250,
                       'order': 6}
    training_mode_list = ['all_in_one'] #['all_in_one', 'leave_one_out', 'one_vs_all']
    constr_k_folds = 10
    model_name_list = ['CNN_PwMhAT','CNN_PwMhATv3','CNN_ReSpAT','ResNet_AT','ConvMixer1D','ML_STIM']#['ML_STIM','CNN_PwMhAT','ConvMixer1D','wideCNN_PwMhAT']#'ResNet_AT', 'ResCNN_SpAT']#['ML_STIM','CNN_PwMhAT','ConvMixer1D','ResCNN_SpAT','ResNet_AT', 'wideCNN_PwMhAT']

    # B. DATA LOADING
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    print('Loading Dataset...')
    # main_path = '//192.168.164.127/public/projects/2024_UniTO_MERs_Classification/2025_XXX_XAI_STN_MERs/Scripts/dataset'
    main_path = 'C:/Users/DETUser/Documents/333338_SCISCENTI_FABRIZIO/dataset_10k_centralCie_testfix'
    # main_path = 'D:/2025_UniTO_MERs_29052025/dataset_10k_anteriorCie'
    all_signals, all_meta = utils.import_dataset.import_signals_and_meta_from_folder(main_path)

    # Import dataset assignment table
    table_path = '//192.168.164.127/public/projects/2024_UniTO_MERs_Classification/2025_XXX_XAI_STN_MERs/Scripts/utils/fold_assignments_cieCentral_testfix.xlsx'
    fold_assignment = pd.read_excel(table_path)
    meta_with_folds = all_meta.merge(fold_assignment, on=['patient', 'dataset'], how='left')

    # B.1 Applying pre-processing
    # -------------------------------------------------------------------------------------
    X_filt = utils.apply_processing.process_data(all_signals, **processing_opts)
    #X_filt = all_signals.copy()
    y = all_meta['class'].values
    y[y > 1] = 0

    del all_signals

    # --------------------------------------------------------------------------------------

    #for test_index in range(5):
    test_index = 2
    val_index = 2

    #for val_index in range(constr_k_folds):
            
    for training_mode in training_mode_list:

        for model_name in model_name_list:

            if model_name != 'ML_STIM':
                print('Applying transformations...')
                X, _ = utils.apply_transform.transform_data(X_filt, fsamp, model_name)
            else:
                X = X_filt.copy()

            now_str = datetime.now().strftime("%d%m%y_%H%M%S")
            #output_folder = f'{training_mode}_training_testing/{model_name}_test{test_index}_val{val_index}_cieMedial_{now_str}'
            output_folder = f'{training_mode}_training_testing/{model_name}_{now_str}'
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            model, lr, bs, scheduler, optimizer, criterion = utils.import_network.import_model(model_name)

            # Save training configuration
            with open(os.path.join(output_folder, 'training_config.txt'), 'w') as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Learning Rate: {lr}\n")
                f.write(f"Batch Size: {bs}\n")
                f.write(f"Scheduler: {scheduler}\n")
                f.write(f"Optimizer: {optimizer}\n")
                f.write(f"Criterion: {criterion}\n")
                f.write(f"Training Mode: {training_mode}\n")
                f.write(f"Train-Val Split (k_fold): {constr_k_folds}\n")
                f.write("\nProcessing Options:\n")
                for key, value in processing_opts.items():
                    f.write(f"  {key}: {value}\n")

            # C. Training and Validation Set definition
            ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

            if training_mode == 'all_in_one':

                # train_mask = meta_with_folds[f'test_3_{constr_k_folds}_fold'].isin(range(0, constr_k_folds-1))
                # val_mask = meta_with_folds[f'test_3_{constr_k_folds}_fold'] == constr_k_folds-1
                # test_mask = meta_with_folds[f'test_3_{constr_k_folds}_fold'] == -1

                train_mask = (meta_with_folds[f'test_{test_index}_{constr_k_folds}fold'] != val_index+1) & (meta_with_folds[f'test_{test_index}_{constr_k_folds}fold'] != -1)
                val_mask = meta_with_folds[f'test_{test_index}_{constr_k_folds}fold'] == val_index+1
                test_mask = meta_with_folds[f'test_{test_index}_{constr_k_folds}fold'] == -1

                # Split X and y accordingly
                X_train, y_train = X[train_mask], y[train_mask]
                X_val, y_val = X[val_mask], y[val_mask]
                X_test, y_test = X[test_mask], y[test_mask]

                if model_name == 'ML_STIM':
                    print('Applying transformations...')
                    X_train, features_extremes = utils.apply_transform.transform_data(X_train, fsamp, model)
                    X_val, _ = utils.apply_transform.transform_data(X_val, fsamp, model, features_extremes=features_extremes)
                    X_test, _ = utils.apply_transform.transform_data(X_test, fsamp, model, features_extremes=features_extremes)
                    np.save(os.path.join(output_folder, 'features_extremes.npy'), features_extremes)

                model, output = utils.training.main_training(X_train, y_train, X_val, y_val, model, lr, bs, scheduler, optimizer, criterion, output_folder, X_test, y_test)
                del X_train, X_val, X_test

                # Extract assignment table  
                assignment = []
                for mask, label in zip([train_mask, val_mask, test_mask], ['train', 'val', 'test']):
                    sub = meta_with_folds.loc[mask, ['patient', 'dataset']].drop_duplicates()
                    sub = sub.assign(assignment=label)
                    assignment.append(sub)
                assignment_table = pd.concat(assignment, ignore_index=True).drop_duplicates()
                assignment_table.to_csv(os.path.join(output_folder, 'data_split.csv'), index=False)

            elif training_mode == 'leave_one_out':

                for dataset_name in ['AlphaMap','NeuroSet','Ciecierski']:

                    output_dir = os.path.join(output_folder,dataset_name)
                    if not os.path.exists(output_dir): os.makedirs(output_dir)

                    train_mask = (meta_with_folds['dataset'] != dataset_name) & (meta_with_folds[f'x{constr_k_folds}_fold_cv'].isin(range(0, constr_k_folds-1)))
                    val_mask = (meta_with_folds['dataset'] != dataset_name) & (meta_with_folds[f'x{constr_k_folds}_fold_cv'] == constr_k_folds-1)
                    test_mask = meta_with_folds['dataset'] == dataset_name

                    # Split X and y accordingly
                    X_train, y_train = X[train_mask], y[train_mask]
                    X_val, y_val = X[val_mask], y[val_mask]
                    X_test, y_test = X[test_mask], y[test_mask]

                    if model_name == 'ML_STIM':
                        print('Applying transformations...')
                        X_train, features_extremes = utils.apply_transform.transform_data(X_train, fsamp, model)
                        X_val, _ = utils.apply_transform.transform_data(X_val, fsamp, model, features_extremes=features_extremes)
                        X_test, _ = utils.apply_transform.transform_data(X_test, fsamp, model, features_extremes=features_extremes)
                        np.save(os.path.join(output_dir, 'features_extremes.npy'), features_extremes)

                    model, output = utils.training.main_training(X_train, y_train, X_val, y_val, model, lr, bs, output_dir, X_test, y_test)
                    del X_train, X_val, X_test

                    # Extract assignment table
                    assignment = []
                    for mask, label in zip([train_mask, val_mask, test_mask], ['train', 'val', 'test']):
                        sub = meta_with_folds.loc[mask, ['patient', 'dataset']].drop_duplicates()
                        sub = sub.assign(assignment=label)
                        assignment.append(sub)
                    assignment_table = pd.concat(assignment, ignore_index=True).drop_duplicates()
                    assignment_table.to_csv(os.path.join(output_dir, 'data_split.csv'), index=False)

            elif training_mode == 'one_vs_all':
                
                for dataset_name in ['AlphaMap','NeuroSet','Ciecierski']:

                    output_dir = os.path.join(output_folder,dataset_name) 
                    if not os.path.exists(output_dir): os.makedirs(output_dir)

                    train_mask = (meta_with_folds['dataset'] == dataset_name) & (meta_with_folds[f'x{constr_k_folds}_fold_cv'].isin(range(0, constr_k_folds-1)))
                    val_mask = (meta_with_folds['dataset'] == dataset_name) & (meta_with_folds[f'x{constr_k_folds}_fold_cv'] == constr_k_folds-1)
                    test_mask = meta_with_folds['dataset'] != dataset_name

                    # Split X and y accordingly
                    X_train, y_train = X[train_mask], y[train_mask]
                    X_val, y_val = X[val_mask], y[val_mask]
                    X_test, y_test = X[test_mask], y[test_mask]

                    if model_name == 'ML_STIM':
                        print('Applying transformations...')
                        X_train, features_extremes = utils.apply_transform.transform_data(X_train, fsamp, model)
                        X_val, _ = utils.apply_transform.transform_data(X_val, fsamp, model, features_extremes=features_extremes)
                        X_test, _ = utils.apply_transform.transform_data(X_test, fsamp, model, features_extremes=features_extremes)
                        np.save(os.path.join(output_dir, 'features_extremes.npy'), features_extremes)

                    model, output = utils.training.main_training(X_train, y_train, X_val, y_val, model, lr, bs, output_dir, X_test, y_test)
                    del X_train, X_val, X_test
            
                    # Extract assignment table
                    assignment = []
                    for mask, label in zip([train_mask, val_mask, test_mask], ['train', 'val', 'test']):
                        sub = meta_with_folds.loc[mask, ['patient', 'dataset']].drop_duplicates()
                        sub = sub.assign(assignment=label)
                        assignment.append(sub)
                    assignment_table = pd.concat(assignment, ignore_index=True).drop_duplicates()
                    assignment_table.to_csv(os.path.join(output_dir, 'data_split.csv'), index=False)
            
            # del X_train, X_val, X_test