"""
Configuration file for MER dataset preparation.

Edit these parameters to match your setup and requirements.
"""

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base path where raw datasets are stored
# Expected structure: BASE_PATH/dataset_name/
BASE_PATH = 'D:/2025_UniTO_MERs_29052025'

# Output path for processed segments
OUTPUT_PATH = 'dataset_processed_windows'

# Output path for organized dataset (train/val/test splits)
ORGANIZED_PATH = 'dataset'

# Path to assignment CSV file
ASSIGNMENT_CSV = 'data_split.csv'

# Path to save computed overlap values
META_OVERLAP_PATH = 'dataset_processed_windows/data_overlap.csv'

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

# Datasets to process
DATASET_NAMES = ['AlphaMap', 'NeuroSet', 'NeuroOmega']

# Sampling frequencies for each dataset (Hz)
SAMPLING_FREQUENCIES = {
    'NeuroOmega': 24000,
    'AlphaMap': 12000,
    'NeuroSet': 20000
}

# =============================================================================
# PREPROCESSING PARAMETERS
# =============================================================================

# Target sampling frequency after resampling (Hz)
TARGET_SAMPLING_FREQ = 10000

# Low-pass filter cutoff frequency (Hz)
FILTER_CUTOFF = 5000

# Filter order (Butterworth)
FILTER_ORDER = 6

# =============================================================================
# SEGMENTATION PARAMETERS
# =============================================================================

# Window size in samples (at TARGET_SAMPLING_FREQ)
# 10000 samples = 1 second at 10kHz
WINDOW_SIZE = 10000

# Minimum overlap (0.0 to 1.0)
# 0.25 = 25% overlap between consecutive windows
MIN_OVERLAP = 0.25

# Maximum overlap (0.0 to 1.0)
# 0.75 = 75% overlap between consecutive windows
MAX_OVERLAP = 0.75

# Overlap adjustment step size
# Used in the iterative overlap optimization
OVERLAP_STEP = 0.05

# =============================================================================
# SPLITTING PARAMETERS
# =============================================================================

# Number of patients per dataset in test set
# The algorithm will try to select this many patients from each dataset
TEST_SIZE = 10

# Random seeds for generating different split options
# Each seed creates a different random split for comparison
RANDOM_SEEDS = [42, 123, 456, 789, 999]

# Number of cross-validation folds to generate
# Creates both 5-fold and 10-fold CV splits
N_FOLDS = [5, 10]

# =============================================================================
# ELECTRODE FILTERING
# =============================================================================

# Allowed electrode labels for recordings
# Only recordings with these electrode labels will be imported
ALLOWED_ELECTRODES = ['Central', 'Electrode1']

# =============================================================================
# FIXED TEST SET (OPTIONAL)
# =============================================================================

# If you want to force specific patients into the test set, uncomment and
# modify the following list. Each tuple is (patient_id, dataset_name).
# If None or empty list, all test patients are selected randomly.

# FIXED_TEST_PAIRS = None

# Example with fixed patients:
FIXED_TEST_PAIRS = [
    ('P12', 'NeuroSet'), ('P011', 'AlphaMap'), ('P04', 'Ciecierski'),
    ('P17', 'NeuroSet'), ('P012', 'AlphaMap'), ('P05', 'Ciecierski'),
    ('P18', 'NeuroSet'), ('P016', 'AlphaMap'), ('P11', 'Ciecierski'),
    ('P20', 'NeuroSet'), ('P026', 'AlphaMap'), ('P14', 'Ciecierski'),
    ('P21', 'NeuroSet'), ('P041', 'AlphaMap'), ('P17', 'Ciecierski'),
    ('P23', 'NeuroSet'), ('P052', 'AlphaMap'), ('P20', 'Ciecierski'),
    ('P27', 'NeuroSet'), ('P069', 'AlphaMap'), ('P25', 'Ciecierski'),
    ('P28', 'NeuroSet'), ('P081', 'AlphaMap'), ('P28', 'Ciecierski'),
    ('P33', 'NeuroSet'), ('P083', 'AlphaMap'), ('P36', 'Ciecierski'),
    ('P35', 'NeuroSet'), ('P085', 'AlphaMap'), ('P41', 'Ciecierski'),
]

# =============================================================================
# ORGANIZATION PARAMETERS
# =============================================================================

# Whether to copy files (True) or move them (False) when assigning to splits 
# (train/val/test)
COPY_FILES = True

# =============================================================================
# ADVANCED OPTIONS
# =============================================================================

# Verbose output during processing
VERBOSE = True

# Number of trials for balanced test set generation
# Higher values may find better balanced splits but take longer
N_BALANCE_TRIALS = 1000

# Tolerance for fold balancing (number of segments)
# Stops adjusting overlap when within this tolerance
FOLD_BALANCE_TOLERANCE = 10
