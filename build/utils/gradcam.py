import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.networks.CNN_PwMhAT import CNN_PwMhAT
import utils
import scipy
import matplotlib
import random

# ---- Grad-CAM Computation ----
def compute_gradcam_for_class(model, class_idx, output, feat_grads, attn_out, input_signal):
	
    # Gradients w.r.t. the target class
    # ----------------------------------
    model.zero_grad()
    score = output[0, class_idx]
    score.backward(retain_graph=True)
    # feat_grads = torch.nn.functional.relu(feat_grads) ##### Grad-CAM+

    # Grad-CAM
    # ----------------------------------
    weights = feat_grads.mean(dim=2, keepdim=True)
    gradcam = (weights * attn_out).sum(dim=2).squeeze(0)
    gradcam = torch.nn.functional.relu(gradcam)
    gradcam = gradcam.cpu().numpy()
    # gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)
    # gradcam = gradcam / np.sum(gradcam) * torch.nn.functional.softmax(output, dim=1)[0, class_idx].detach().cpu().numpy()

    # Upsampling
    # ----------------------------------
    upsampled = np.repeat(gradcam, input_signal.shape[-1] // gradcam.shape[0])
    if upsampled.shape[0] < input_signal.shape[-1]:
        upsampled = np.pad(upsampled, (0, input_signal.shape[-1] - upsampled.shape[0]), mode='edge')
    elif upsampled.shape[0] > input_signal.shape[-1]:
        upsampled = upsampled[:input_signal.shape[-1]]
    # upsampled = scipy.signal.resample(gradcam, input_signal.shape[-1])

    upsampled = upsampled / np.sum(upsampled) * torch.nn.functional.softmax(output, dim=1)[0, class_idx].detach().cpu().numpy()

    return upsampled

def compute_attention_vec(attn_map, input_signal):
    # Attention vector
    # ----------------------------------
    attn_vec = np.sum(attn_map, axis=1).squeeze()

    # Upsampling
    # ----------------------------------
    attn_num_patches = attn_vec.shape[0]
    attn_input_len = input_signal.shape[-1]
    upsampled_attn_vec = np.repeat(attn_vec, attn_input_len // attn_num_patches)
    if upsampled_attn_vec.shape[0] < attn_input_len:
        upsampled_attn_vec = np.pad(upsampled_attn_vec, (0, attn_input_len - upsampled_attn_vec.shape[0]), mode='edge')
    elif upsampled_attn_vec.shape[0] > attn_input_len:
        upsampled_attn_vec = upsampled_attn_vec[:attn_input_len]

    upsampled_attn_vec = upsampled_attn_vec / np.sum(upsampled_attn_vec)

    return upsampled_attn_vec

def moving_average(x, w):
    t = np.arange(0, len(x))
    # x_smooth = np.convolve(x, np.ones(w)/w, mode='same')
    x_smooth = scipy.ndimage.gaussian_filter1d(x, sigma=w)
    x_smooth = x_smooth / x_smooth.max() * x.max()
    return x_smooth, t

def plot_signal_and_gradcam(input_signal, output):

    upsampled_gradcam_0 = output['gradcam_0']
    upsampled_gradcam_0_ma = output['gradcam_0_ma']
    upsampled_gradcam_1 = output['gradcam_1']
    upsampled_gradcam_1_ma = output['gradcam_1_ma']
    upsampled_attn_vec = output['attn_vec']
    upsampled_attn_vec_ma = output['attn_vec_ma']
    t = output['time']
    fsamp = output['fsamp']

    fig = plt.figure(figsize=(15, 10))
    gs = matplotlib.gridspec.GridSpec(3, 2, width_ratios=[40, 1], height_ratios=[1, 1, 1])

    # Create the first axis
    ax1 = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    # Create ax2 and ax3 sharing x-axis with ax1
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

    m = np.min(np.concatenate((upsampled_gradcam_0_ma, upsampled_gradcam_1_ma)))
    M = np.max(np.concatenate((upsampled_gradcam_0_ma, upsampled_gradcam_1_ma)))
    norm_gradcam = (upsampled_gradcam_1_ma - m) / (M - m + 1e-8)
    points = np.array([t, input_signal[0,:]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = matplotlib.collections.LineCollection(segments, cmap='berlin', norm=plt.Normalize(0, 1))
    lc.set_array(norm_gradcam)
    lc.set_linewidth(1)
    ax1.add_collection(lc)
    ax1.set_xlim(t.min(), t.max())
    ax1.set_ylim(input_signal[0].min(), input_signal[0].max())
    ax1.set_title('Input Signal')
    ax1.set_ylabel('normalized amplitude (a.u.)')
    cbar = plt.colorbar(lc, cax=cax)
    cbar.ax.set_yticklabels([])  # Hide tick labels
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['LOW', 'HIGH'])
    cbar.set_label('STN probability', labelpad=0)

    ax2.plot(upsampled_gradcam_0, color='tab:blue', alpha=0.3)
    ax2.plot(upsampled_gradcam_1, color='tab:orange', alpha=0.3)
    ax2.plot(upsampled_gradcam_0_ma, color='blue', label='Smoothed Grad-CAM Class 0', linewidth=2)
    ax2.plot(upsampled_gradcam_1_ma, color='orange', label='Smoothed Grad-CAM Class 1', linewidth=2)
    ax2.set_ylabel('relative contribution (a.u.)')
    ax2.legend()

    ax3.plot(upsampled_attn_vec, color='tab:green', alpha=0.3)
    ax3.plot(upsampled_attn_vec_ma, color='green', label='Smoothed Attention', linewidth=2)
    ax3.set_ylabel('relative attention (a.u.)')
    ax3.legend()

    # Optionally hide x-axis labels for ax1 and ax2 to avoid overlap
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.show()

def grad_cam(model,input_signal,to_plot=True):

    PROCESSING_OPTS = {'fsamp': 10000, 
                       'normalization': 'zscore', 
                       'filter_type': ['highpass', 'notch'], 
                       'notch_freqs': np.arange(100, 4000, 50), 
                       'lowcut': 250, 'order': 6}

    # Grad-CAM Hooks
    # ----------------------------------
    def save_attn_out(module, input, output):
        global attn_out
        attn_out = output[0].detach()  # attn_out from self-attention
    def save_attn_grad(module, grad_input, grad_output):
        global feat_grads
        feat_grads = grad_output[0].detach()

    # Declare globals for hooks
    global feat_grads, attn_out
    feat_grads = None
    attn_out = None
    # Hooks are registered on the 'attention' module of the model to capture features and gradients
    handle_fwd = model.attention.register_forward_hook(lambda m, inp, out: save_attn_out(m, inp, out))
    handle_bwd = model.attention.register_full_backward_hook(lambda m, gin, gout: save_attn_grad(m, gin, gout))

    # Inference
    # ----------------------------------
    # varMask = utils.apply_transform.get_var_mask(input_signal, PROCESSING_OPTS['fsamp'])
    # _, artMask = utils.apply_transform.remove_artifacts_with_replacement_v2(input_signal, PROCESSING_OPTS['fsamp'])
    input_signal = utils.apply_processing.process_data(input_signal, verbose=False, **PROCESSING_OPTS)

    input_dim = 10000
    N_windows = input_signal.shape[-1] // input_dim
    input_signal = input_signal[:,:N_windows*input_dim]  # Trim to multiple of input_dim

    gradcam_0 = []
    gradcam_1 = []
    attn_vec = []
    scores = []

    device = next(model.parameters()).device

    for i in range(N_windows):

        input_signal_window = input_signal[0,i*input_dim:(i+1)*input_dim]
        input_tensor = torch.tensor(input_signal_window, dtype=torch.float32).unsqueeze(0).to(device)

        # Forward pass
        score, aux = model(input_tensor)
        pred_class = score.argmax(dim=1).item()
        output = torch.nn.functional.softmax(score, dim=1)
        
        ups_out = output.detach().cpu().numpy()
        ups_out = np.repeat(ups_out, input_signal_window.shape[-1], axis=0)
        if ups_out.shape[0] < input_signal_window.shape[-1]:
            ups_out = np.pad(ups_out, (0, input_signal_window.shape[-1] - ups_out.shape[0]), mode='edge')
        elif ups_out.shape[0] > input_signal_window.shape[-1]:
            ups_out = ups_out[:input_signal_window.shape[-1]]
        scores.append(ups_out)

        # Grad-CAM Computation for class 0
        # ----------------------------------
        model.zero_grad()
        score[0, 0].backward(retain_graph=True)
        upsampled_gradcam_0 = compute_gradcam_for_class(model, 0, score, feat_grads, attn_out, input_signal_window)
        gradcam_0.append(upsampled_gradcam_0)

        # Grad-CAM Computation for class 1
        # ----------------------------------
        model.zero_grad()
        score[0, 1].backward(retain_graph=True)
        upsampled_gradcam_1 = compute_gradcam_for_class(model, 1, score, feat_grads, attn_out, input_signal_window)
        gradcam_1.append(upsampled_gradcam_1)

        # Attention Map
        # ----------------------------------
        attn_map = aux['attention_map'].detach().cpu().numpy().squeeze(0)
        upsampled_attn_vec = compute_attention_vec(attn_map, input_signal_window)
        attn_vec.append(upsampled_attn_vec)

    gradcam_0 = np.concatenate(gradcam_0)
    gradcam_1 = np.concatenate(gradcam_1)
    attn_vec = np.concatenate(attn_vec)
    scores = np.concatenate(scores)

    gradcam_0_ma, t = moving_average(gradcam_0, w=500)
    gradcam_1_ma, _ = moving_average(gradcam_1, w=500)
    attn_vec_ma, _ = moving_average(attn_vec, w=500)

    # ---- Cleanup ----
    handle_fwd.remove()
    handle_bwd.remove()

    output = {
        'output_signal': input_signal,
        'gradcam_0': gradcam_0,
        'gradcam_0_ma': gradcam_0_ma,
        'gradcam_1': gradcam_1,
        'gradcam_1_ma': gradcam_1_ma,
        'attn_vec': attn_vec,
        'attn_vec_ma': attn_vec_ma,
        'time': t,
        'fsamp': PROCESSING_OPTS['fsamp'],
        'scores': scores
    }

    if to_plot:
        plot_signal_and_gradcam(input_signal, output)

    return output