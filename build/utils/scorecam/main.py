"""
ScoreCAM implementation for CNN_PwMhAT model on 1D signals.
Adapted from the original ScoreCAM for 2D images to work with 1D time-series data.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn.functional as F
import numpy as np
import os
from CNN_PwMhAT import CNN_PwMhAT
from CNN_PwMhATv3 import CNN_PwMhATv3
from scorecam1D import ScoreCAM1D
from scipy import stats


def load_model(model_path, device='cuda'):
    """
    Load the CNN_PwMhAT model with trained weights.
    
    Args:
        model_path: Path to the saved model weights (.pt file)
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        model: Loaded model in eval mode
    """
    # Initialize model with same parameters as training
    # model = CNN_PwMhAT(
    #     input_size=10000,
    #     patch_size=200,
    #     in_channels=1,
    #     out_channels=32,
    #     embed_dim=32,
    #     num_heads=2,
    #     output_dim=2,
    #     dropout=0.1
    # )
    model = CNN_PwMhATv3(input_length=10000)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model


def load_data(data_path, fsamp=20000, fsamp_new=10000):
    """
    Load data from .npz file.
    
    Args:
        data_path: Path to the .npz file
        
    Returns:
        data: numpy array with signals (each row is a signal)
    """
    npz_data = np.load(data_path)
    
    # Print available keys in the npz file
    print(f"Available keys in {data_path}: {list(npz_data.keys())}")
    
    # Try to get the data array (assuming it's stored under a common key)
    if 'data' in npz_data:
        data = npz_data['data']
    elif 'X' in npz_data:
        data = npz_data['X']
    elif 'signals' in npz_data:
        data = npz_data['signals']
    else:
        # Get the first array in the file
        key = list(npz_data.keys())[0]
        data = npz_data[key]
        print(f"Using key '{key}' from npz file")

    data = data[:,:10*fsamp]
    
    from scipy.signal import resample
    if fsamp != fsamp_new:
        num_samples = int(data.shape[1] * fsamp_new / fsamp)
        data_resampled = []
        for signal in data:
            resampled_signal = resample(signal, num_samples)
            data_resampled.append(resampled_signal)
        data = np.array(data_resampled)
        print(f"Data resampled from {fsamp} Hz to {fsamp_new} Hz")
    
    print(f"Data shape: {data.shape}")
    return data


def classify_and_compute_scorecam(model, data, device='cuda', target_layer_name='patch_encoder'):
    """
    Classify signals and compute ScoreCAM for each signal.
    
    Args:
        model: Loaded CNN_PwMhAT model
        data: numpy array of signals (n_samples, signal_length)
        device: Device to run on
        target_layer_name: Name of the layer to compute ScoreCAM on
        
    Returns:
        predictions: Array of predicted classes
        probabilities: Array of prediction probabilities
        scorecam_maps: List of ScoreCAM saliency maps
    """
    # Get the target layer
    target_layer = getattr(model, target_layer_name)
    
    # Initialize ScoreCAM
    scorecam = ScoreCAM1D(model, target_layer)
    
    predictions = []
    probabilities = []
    scorecam_maps = []
    
    print(f"\nProcessing {len(data)} signals...")

    import apply_processing
    PROCESSING_OPTS = {'fsamp': 10000, 
                       'normalization': 'zscore', 
                       'filter_type': ['highpass', 'notch'], 
                       'notch_freqs': np.arange(250, 3000, 50), 
                       'lowcut': 200, 'order': 6}
    data = apply_processing.process_data(data, verbose=False, **PROCESSING_OPTS)
    
    for idx, signal in enumerate(data):
        # Lists to store results for all windows in this signal
        signal_predictions = []
        signal_probabilities = []
        signal_scorecam_maps = []
        
        # Window parameters
        win_len = 10000
        num_windows = len(signal) // win_len
        
        print(f"\nProcessing signal {idx+1}/{len(data)} - {num_windows} windows")
        
        # Process each window in the signal
        for win_idx in range(num_windows):
            # Extract window
            window = signal[win_idx * win_len:(win_idx + 1) * win_len]
            window_tensor = torch.from_numpy(window).float().unsqueeze(0).to(device)  # (1, length)
            
            # Get prediction
            with torch.no_grad():
                output, _ = model(window_tensor)
                probs = F.softmax(output, dim=1)
                pred_class = output.argmax(dim=1).item()
                #pred_prob = probs[0, pred_class].item()
                pred_prob = probs[0, 1].item()  # Probability of class 1

            signal_predictions.append(pred_class)
            signal_probabilities.append(pred_prob)
            
            #scorecam_map = scorecam(window_tensor, class_idx=pred_class)
            scorecam_map = scorecam(window_tensor, class_idx=1)
            
            if scorecam_map is not None:
                # signal_scorecam_maps.append(scorecam_map.cpu().numpy())
                signal_scorecam_maps.append(scorecam_map.cpu().numpy() * pred_prob)
            else:
                signal_scorecam_maps.append(np.zeros((1, 1, win_len)))
        
        # Append results for this signal to the main lists
        predictions.append(np.array(signal_predictions))
        probabilities.append(np.array(signal_probabilities))
        scorecam_maps.append(np.array(signal_scorecam_maps))
    
    return predictions, probabilities, scorecam_maps


def main():
    """
    Main function to load model, data, classify signals and compute ScoreCAM.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Paths
    # model_path = 'best_model.pt'
    # data_path = 'P18_LEFT.npz'
    model_path = 'best_model_v3.pt'
    data_path = 'P18_LEFT.npz'
    
    # Load model
    print("Loading model...")
    model = load_model(model_path, device)
    
    # Load data
    print("\nLoading data...")
    data = load_data(data_path, fsamp=20000, fsamp_new=10000)
    
    # Classify and compute ScoreCAM
    print("\nClassifying signals and computing ScoreCAM...")
    # target_layer_name = 'patch_encoder'
    # target_layer_name = 'attention'
    target_layer_name = 'encoder'
    predictions, probabilities, scorecam_maps = classify_and_compute_scorecam(
        model, data, device, target_layer_name
    )

    # ------------------------------------------
    import matplotlib.pyplot as plt
    for idx, signal in enumerate(data):
        # Prepare input
        signal = (signal - signal.mean()) / (15*data.std())
        map = np.squeeze(scorecam_maps[idx])
        map = map.flatten()  # or map.ravel() - concatenates all segments into (N*10000,)
        
        import scipy
        map = scipy.ndimage.gaussian_filter1d(map, sigma=500)
        
        plt.imshow(map[np.newaxis, :], extent=[0, len(signal), -idx-.5, -idx+.5], aspect='auto', cmap='coolwarm', alpha=0.6, origin='lower')
        plt.plot(signal-idx,color='black',linewidth=0.8)
        #plt.plot(map/(map.max())/2 - idx,color='red',linewidth=1)
        plt.plot(map/2-idx,color='red',linewidth=1)

    plt.ylim(-len(data)+1-.6, .6)
    plt.show()
    # ------------------------------------------
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for i in range(len(predictions)):
        print(f"Signal {i+1}: Class={stats.mode(predictions[i]).mode}, Probability={np.mean(probabilities[i]):.4f}, "
              f"ScoreCAM={'Available' if scorecam_maps[i] is not None else 'None'}")
    
    # Save results
    output_path = 'scorecam_results.npz'
    np.savez(
        output_path,
        predictions=predictions,
        probabilities=probabilities,
        scorecam_maps=np.array(scorecam_maps, dtype=object)
    )
    print(f"\nResults saved to {output_path}")
    
    return predictions, probabilities, scorecam_maps


if __name__ == "__main__":
    predictions, probabilities, scorecam_maps = main()
