import sys
from pathlib import Path
import mne
import numpy as np
import warnings

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    EEG_CHANNELS, EOG_CHANNELS, ECG_CHANNELS, EMG_CHANNELS,
    NOTCH_FREQS, HP_FREQ, LP_FREQ,
    EPOCH_DURATION, EPOCH_OVERLAP, REJECT_PEAK_TO_PEAK
)

# Suppress verbose mne info when desired
mne.set_log_level("WARNING")

def load_and_setup_raw(bdf_path):
    """
    Loads BDF file, assigns channel types (EEG, EOG, ECG, EMG),
    and applies standard 10-20 montage to EEG channels.
    """
    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
    
    # Map channel types
    channel_types = {}
    for ch in raw.ch_names:
        if ch in EEG_CHANNELS:
            channel_types[ch] = "eeg"
        elif ch in EOG_CHANNELS:
            channel_types[ch] = "eog"
        elif ch in ECG_CHANNELS:
            channel_types[ch] = "ecg"
        elif ch in EMG_CHANNELS:
            channel_types[ch] = "emg"
        elif "status" in ch.lower():
            channel_types[ch] = "stim"
            
    raw.set_channel_types(channel_types)
    
    # Apply standard 10-20 montage
    montage = mne.channels.make_standard_montage("colin27_1020")
    raw.set_montage(montage, on_missing="ignore", match_case=False)
    return raw

def filter_signal(raw):
    """
    Applies 50 Hz (+100 Hz) notch filter and 0.5-45 Hz bandpass filter.
    """
    raw_filtered = raw.copy()
    # Notch filter for powerline noise
    raw_filtered.notch_filter(freqs=NOTCH_FREQS, picks=["eeg", "eog", "ecg"], verbose=False)
    # Bandpass filter
    raw_filtered.filter(l_freq=HP_FREQ, h_freq=LP_FREQ, picks=["eeg", "eog", "ecg"], verbose=False)
    return raw_filtered

def clean_artifacts_ica(raw, random_state=42):
    """
    Runs FastICA to identify and remove ocular and cardiac artifacts
    using correlation with EOG and ECG channels.
    Downsamples to 125 Hz for 4x faster and more stable ICA fitting.
    """
    # Downsample and high-pass for fast, robust ICA fitting
    picks_ica = [ch for ch in raw.ch_names if ch in EEG_CHANNELS + EOG_CHANNELS + ECG_CHANNELS]
    raw_ica_prep = raw.copy().pick(picks_ica)
    raw_ica_prep.resample(125.0, verbose=False)
    raw_ica_prep.filter(l_freq=1.0, h_freq=40.0, picks="eeg", verbose=False)
    
    n_components = min(15, len(EEG_CHANNELS))
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="fastica",
        max_iter="auto",
        rng=random_state
    )
    ica.fit(raw_ica_prep, picks="eeg", verbose=False)
    
    # Find EOG artifacts if EOG channels exist
    eog_indices = []
    if any(ch in raw.ch_names for ch in EOG_CHANNELS):
        for eog_ch in [ch for ch in EOG_CHANNELS if ch in raw.ch_names]:
            inds, _ = ica.find_bads_eog(raw, ch_name=eog_ch, threshold=3.0, verbose=False)
            eog_indices.extend(inds)
            
    # Find ECG artifacts if ECG channel exists
    ecg_indices = []
    if any(ch in raw.ch_names for ch in ECG_CHANNELS):
        for ecg_ch in [ch for ch in ECG_CHANNELS if ch in raw.ch_names]:
            inds, _ = ica.find_bads_ecg(raw, ch_name=ecg_ch, method="correlation", threshold="auto", verbose=False)
            ecg_indices.extend(inds)
            
    # Combine bad components (unique)
    exclude_components = list(set(eog_indices + ecg_indices))
    ica.exclude = exclude_components
    
    # Apply ICA cleaning to raw signal
    raw_cleaned = raw.copy()
    ica.apply(raw_cleaned, verbose=False)
    
    return raw_cleaned, ica, exclude_components

def rereference_and_select_eeg(raw):
    """
    Sets Common Average Reference (CAR) and retains only the 26 scalp EEG channels.
    """
    # Pick only scalp EEG channels
    raw_eeg = raw.copy().pick(EEG_CHANNELS)
    # Common Average Reference
    raw_eeg.set_eeg_reference("average", projection=False, verbose=False)
    return raw_eeg

def segment_into_epochs(raw_eeg):
    """
    Segments continuous EEG into fixed duration epochs with overlap,
    and applies peak-to-peak rejection threshold.
    """
    events = mne.make_fixed_length_events(
        raw_eeg,
        duration=EPOCH_DURATION * (1.0 - EPOCH_OVERLAP),  # Step size
        start=0.0
    )
    
    reject_criteria = dict(eeg=REJECT_PEAK_TO_PEAK)
    epochs = mne.Epochs(
        raw_eeg,
        events,
        tmin=0.0,
        tmax=EPOCH_DURATION,
        baseline=None,
        reject=reject_criteria,
        preload=True,
        verbose=False
    )
    # If all epochs were dropped due to threshold, retry with relaxed threshold
    if len(epochs) == 0:
        epochs = mne.Epochs(
            raw_eeg,
            events,
            tmin=0.0,
            tmax=EPOCH_DURATION,
            baseline=None,
            reject=dict(eeg=300e-6),  # Relaxed threshold
            preload=True,
            verbose=False
        )
        if len(epochs) == 0:
            # Keep all epochs if still zero
            epochs = mne.Epochs(
                raw_eeg,
                events,
                tmin=0.0,
                tmax=EPOCH_DURATION,
                baseline=None,
                reject=None,
                preload=True,
                verbose=False
            )
    return epochs

def preprocess_eeg_subject(bdf_path, apply_ica=True):
    """
    Complete end-to-end preprocessing for a single BDF recording.
    Returns cleaned Epochs object and summary metadata.
    """
    raw = load_and_setup_raw(bdf_path)
    raw_filt = filter_signal(raw)
    
    bad_components = []
    if apply_ica:
        try:
            raw_clean, ica, bad_components = clean_artifacts_ica(raw_filt)
        except Exception as e:
            warnings.warn(f"ICA failed with error: {e}. Falling back to uncleaned raw.")
            raw_clean = raw_filt
    else:
        raw_clean = raw_filt
        
    raw_car = rereference_and_select_eeg(raw_clean)
    epochs = segment_into_epochs(raw_car)
    
    stats = {
        "duration_sec": raw.times[-1],
        "n_epochs_retained": len(epochs),
        "ica_bad_components": len(bad_components),
        "channels": epochs.ch_names
    }
    return epochs, stats
