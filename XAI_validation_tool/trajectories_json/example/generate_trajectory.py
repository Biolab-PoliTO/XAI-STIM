import json
import numpy as np

# Generate example trajectory with 8 channels, 50000 samples (5s @ 10kHz)
np.random.seed(42)
num_channels = 8
num_samples = 10000
sampling_rate = 2000

# Generate realistic-looking STN signals
signals = []
for i in range(num_channels):
    # Base signal with some variation
    base = np.random.randn(num_samples) * 50
    # Add high-frequency component for STN-like activity in middle channels
    if 2 <= i <= 5:
        hf_start = int(num_samples * 0.3)
        hf_end = int(num_samples * 0.7)
        base[hf_start:hf_end] += np.random.randn(hf_end - hf_start) * 150
    signals.append(base.tolist())

# Generate scorecam heatmap (downsampled for visualization, e.g., 1000 points)
scorecam_samples = 1000
scorecam = []
for i in range(num_channels):
    heatmap_row = np.zeros(scorecam_samples)
    # Higher values in middle channels where STN activity is
    if 2 <= i <= 5:
        # Gaussian-like activation in middle time period
        center = scorecam_samples // 2
        for j in range(scorecam_samples):
            dist = abs(j - center) / (scorecam_samples * 0.2)
            heatmap_row[j] = max(0, 1.0 - dist**2) * (1 - abs(i - 3.5) / 2)
    scorecam.append(heatmap_row.tolist())

trajectory = {
    'id': 'T01',
    'name': 'Trajectory 01 - Patient P001 Right',
    'sampling_rate': sampling_rate,
    'signals': signals,
    'depths': [8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0],
    'classification': {
        'label': 'STN',
        'confidence': 0.87
    },
    'channel_probabilities': [0.32, 0.41, 0.58, 0.72, 0.81, 0.69, 0.54, 0.38],
    'scorecam': scorecam
}

with open('html_version/trajectory_T01.json', 'w') as f:
    json.dump(trajectory, f, indent=2)

print(f'Created trajectory_T01.json with {num_channels} channels and {num_samples} samples')
