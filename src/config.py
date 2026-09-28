import os
from pathlib import Path

# Paths
BASE_DIR = Path("d:/ADHD_Project")
DATASET_DIR = BASE_DIR / "TDBRAIN_Dataset_V3_1_Encr" / "TDBRAIN_Dataset_V3_1"
METADATA_SOURCE = Path(r"C:\Users\navan\Downloads\TDBRAIN_participants_V3.xlsx")
METADATA_DIR = BASE_DIR / "data" / "metadata"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

# Channels
EEG_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC3", "FCz", "FC4", "T7", "C3", "Cz", "C4", "T8",
    "CP3", "CPz", "CP4", "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2"
]

EOG_CHANNELS = ["VPVA", "VNVB", "HPHL", "HNHR"]
ECG_CHANNELS = ["Erbs"]
EMG_CHANNELS = ["Mass"]

# Filtering parameters
SFREQ_TARGET = 500.0  # Original is 500 Hz
NOTCH_FREQS = [50.0, 100.0]  # Line noise and harmonics
HP_FREQ = 0.5  # High-pass filter cutoff
LP_FREQ = 45.0  # Low-pass filter cutoff

# Epoching parameters
EPOCH_DURATION = 2.0  # Seconds
EPOCH_OVERLAP = 0.5   # 50% overlap (1.0 sec step)
REJECT_PEAK_TO_PEAK = 150e-6  # 150 microvolts rejection threshold

# Frequency bands
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0)
}

# Cohort Filter Settings (Option A)
AGE_MIN = 18.0
AGE_MAX = 60.0  # Match ADHD adult upper bound (~56) to prevent senior control bias
