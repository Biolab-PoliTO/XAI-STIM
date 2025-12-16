import numpy as np
import pandas as pd
import os
import torch
import utils
import time
import glob

def inference(model, X_test, y_test, test_meta, fsamp, processing_opts, features_extremes, output_path):
    # Optimizer and loss
    bs = 10
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    b, a = utils.apply_processing.initialize_filter_coefficients(fsamp=processing_opts['fsamp'],
                                                                filter_type=processing_opts['filter_type'],
                                                                notch_freqs=processing_opts['notch_freqs'],
                                                                lowcut=processing_opts['lowcut'],
                                                                order=processing_opts['order'])

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    times = []
    probs = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            start_time = time.time()
            inputs, labels = inputs.to(device), labels.to(device)

            inputs = utils.apply_processing.process_data(inputs.cpu().numpy().squeeze(), b=b, a=a, verbose=False, **processing_opts)
            inputs, _ = utils.apply_transform.transform_data(inputs, fsamp, model_name, verbose=False, features_extremes=features_extremes)
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)

            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            probababilities = torch.nn.functional.softmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            end_time = time.time()
            times.append([(end_time - start_time)/len(labels)]*len(labels))
            probs.append(probababilities.cpu().numpy())
            print(f"\rProcessed batch {total // bs + 1} / {len(test_loader)}",end='', flush=True)  
        print()

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total

    # Tabulate results and save to CSV, including metadata
    times = np.concatenate(times)
    probs = np.concatenate(probs)
    results_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': np.argmax(probs, axis=1),
        'prob_class_0': probs[:, 0],
        'prob_class_1': probs[:, 1],
        'inference_time': times
    })

    # Concatenate metadata columns
    results_df = pd.concat([test_meta.reset_index(drop=True), results_df], axis=1)

    results_df.to_csv(output_path, index=False)

    print('-'*80)
    print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    print(f"Results saved to {output_path}")
    print('='*80)
    return results_df

if __name__ == "__main__":

    # A. INITIALIZATIONS
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    fsamp = 10000
    processing_opts = {'fsamp': fsamp, 
                       'normalization': 'zscore',
                       'filter_type': ['highpass','notch'],
                       'notch_freqs': np.arange(250, 3000, 50), #np.arange(100, 4000, 50),
                       'lowcut':200, #250,
                       'order': 6}
    main_path = '//192.168.164.127/public/projects/2024_UniTO_MERs_Classification/2025_XXX_XAI_STN_MERs/Scripts/all_in_one_training_testing'

    # B. DATA LOADING
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    print('Loading Dataset...')
    # data_path = '//192.168.164.127/public/projects/2024_UniTO_MERs_Classification/2025_XXX_XAI_STN_MERs/Scripts/dataset'
    # data_path = 'C:/Users/DETUser/Documents/333338_SCISCENTI_FABRIZIO/dataset'
    
    data_path = 'C:/Users/DETUser/Documents/333338_SCISCENTI_FABRIZIO/dataset_10k_centralCie_testfix'
    #data_path = 'G:/2025_UniTO_MERs_29052025/dataset_10k_centralCie'
    all_signals, all_meta = utils.import_dataset.import_signals_and_meta_from_folder(data_path)

    # C.MODEL INITIALIZATION & INFERENCE
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    trained_models = [d for d in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, d))]
    trained_models = [m for m in trained_models if "_" in m]

    for model_path in trained_models:

        output_dir = os.path.join(main_path, model_path, 'inference_results')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not os.path.exists(os.path.join(output_dir, 'test.csv')):# or \
           #not os.path.exists(os.path.join(output_dir, 'train.csv')) or \
           #not os.path.exists(os.path.join(output_dir, 'val.csv')):

            # Import Model
            # ---------------------------------------------------------------------------------

            # if 'ConvMixer1D' in model_path:
            #     model_name = 'ConvMixer1D'
            #     model, _, _, _, _, _ = utils.import_network.import_model(model_name)
            # else:
            #     try:
            #         model_name = '_'.join(model_path.split('_')[:-5])
            #         model, _, _, _, _, _ = utils.import_network.import_model(model_name)
            #     except:
            #         model_name = '_'.join(model_path.split('_')[:2])
            #         model, _, _, _, _, _ = utils.import_network.import_model(model_name)

            if 'ConvMixer1D' in model_path: model_name = 'ConvMixer1D'
            elif 'ML_STIM' in model_path: model_name = 'ML_STIM'
            elif 'CNN_PwMhATv3' in model_path: model_name = 'CNN_PwMhATv3'
            elif 'CNN_PwMhATv2' in model_path: model_name = 'CNN_PwMhATv2'
            elif 'CNN_PwMhAT' in model_path: model_name = 'CNN_PwMhAT'
            elif 'ResNet_AT' in model_path: model_name = 'ResNet_AT'
            elif 'CNN_ReSpAT' in model_path: model_name = 'CNN_ReSpAT'
            else: model_name = '_'.join(model_path.split('_')[:-5])

            try:
                
                model, _, _, _, _, _ = utils.import_network.import_model(model_name)

                try:
                    best_model_path_list = glob.glob(os.path.join(main_path, model_path, '*', '*.pt'))[0]
                except:
                    best_model_path_list = glob.glob(os.path.join(main_path, model_path, '*', '*', '*.pt'))

                if isinstance(best_model_path_list, str): best_model_path_list = [best_model_path_list]
                
                for best_model_path in best_model_path_list:

                    #try:

                    # Change output_dir if best_model_path contains specific keywords
                    output_path = None
                    for keyword in ["AlphaMap", "Ciecierski", "NeuroSet"]:
                        if keyword in best_model_path:
                            output_path = os.path.join(output_dir, keyword)
                            if not os.path.exists(output_path):
                                os.makedirs(output_path)
                            break
                    if output_path is None:
                        output_path = output_dir

                    if os.path.exists(best_model_path):
                        model.load_state_dict(torch.load(best_model_path, weights_only=True))
                        print(f"Best model loaded from {best_model_path}")
                    else:
                        print(f"No best model found at {best_model_path}. Exiting.")
                        exit()

                    features_extremes_path = glob.glob(os.path.join(os.path.dirname(os.path.dirname(best_model_path)), '*.npy'))
                    if features_extremes_path:
                        features_extremes = np.load(features_extremes_path[0], allow_pickle=True)
                        print('Loaded features extremes (from training set)')
                    else:
                        features_extremes = None

                    # Define test set
                    # ---------------------------------------------------------------------------------

                    split_table_path = os.path.join(os.path.dirname(os.path.dirname(best_model_path)), 'data_split.csv')
                    split_table = pd.read_csv(split_table_path)
                    meta_with_folds = all_meta.merge(split_table, on=['patient', 'dataset'], how='left')
                    test_mask = meta_with_folds['assignment'] == 'test'
                    train_mask = meta_with_folds['assignment'] == 'train'
                    val_mask = meta_with_folds['assignment'] == 'val'

                    X_test = all_signals[test_mask]
                    test_meta = meta_with_folds[test_mask]
                    y_test = test_meta['class'].values
                    y_test[y_test > 1] = 0

                    # # Save X_test and test_meta to CSV for inspection (saved per-model in output_path)
                    # # x_test_path = os.path.join(output_path, 'X_test.csv')
                    # # test_meta_path = os.path.join(output_path, 'test_meta.csv')

                    # # Ensure output directory exists (should already, but be safe)
                    # if not os.path.exists(output_path):
                    #     try:
                    #         os.makedirs(output_path, exist_ok=True)
                    #     except Exception as e:
                    #         print(f"Could not create output directory {output_path}: {e}")

                    # if not os.path.exists(x_test_path):
                    #     try:
                    #         # X_test is assumed to be a 2D numpy array (N x 10000)
                    #         # Use savetxt to avoid building a giant DataFrame in memory.
                    #         np.savetxt(x_test_path, X_test, delimiter=',')
                    #         print(f"Saved X_test to {x_test_path}")
                    #     except Exception as e:
                    #         print(f"Failed to save X_test to {x_test_path}: {e}")
                    # else:
                    #     print(f"X_test CSV already exists at {x_test_path}")

                    # if not os.path.exists(test_meta_path):
                    #     try:
                    #         # test_meta is a pandas DataFrame; save directly
                    #         test_meta.reset_index(drop=True).to_csv(test_meta_path, index=False)
                    #         print(f"Saved test_meta to {test_meta_path}")
                    #     except Exception as e:
                    #         print(f"Failed to save test_meta to {test_meta_path}: {e}")
                    # else:
                    #     print(f"test_meta CSV already exists at {test_meta_path}")

                    X_train = all_signals[train_mask]
                    train_meta = meta_with_folds[train_mask]
                    y_train = train_meta['class'].values
                    y_train[y_train > 1] = 0

                    X_val = all_signals[val_mask]
                    val_meta = meta_with_folds[val_mask]
                    y_val = val_meta['class'].values
                    y_val[y_val > 1] = 0

                    # Inference
                    # ---------------------------------------------------------------------------------

                    if os.path.exists(os.path.join(output_path, 'test.csv')):
                        #continue
                        print('test ok')
                    else:
                        test_results = inference(model, X_test, y_test, test_meta, fsamp, processing_opts, features_extremes, os.path.join(output_path, 'test.csv'))

                    # if os.path.exists(os.path.join(output_path, 'train.csv')):
                    #     #continue
                    #     print('train ok')
                    # else:
                    #     train_results = inference(model, X_train, y_train, train_meta, fsamp, processing_opts, features_extremes, os.path.join(output_path, 'train.csv'))

                    # if os.path.exists(os.path.join(output_path, 'val.csv')):
                    #     print('val ok')
                    #     #continue
                    # else:
                    #     val_results = inference(model, X_val, y_val, val_meta, fsamp, processing_opts, features_extremes, os.path.join(output_path, 'val.csv'))
                    
                    #except:
                    #    continue
            except:
                continue



    