import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import simpson

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import BANDS

def compute_band_powers(epochs):
    """
    Computes absolute and relative spectral band powers using Welch's PSD across epochs.
    Also calculates fronto-central Theta/Beta Ratio (TBR).
    
    Returns a dictionary of aggregated feature values for the subject recording.
    """
    # Compute PSD using Welch's method via MNE's compute_psd
    # Frequency range 0.5 to 45 Hz
    spectrum = epochs.compute_psd(method="welch", fmin=0.5, fmax=45.0, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)
    # psds shape: (n_epochs, n_channels, n_freqs)
    # Average across epochs
    psds_mean = np.mean(psds, axis=0)  # (n_channels, n_freqs)
    
    ch_names = epochs.ch_names
    features = {}
    
    # Total power across all bands for relative power calculation
    total_power_per_ch = simpson(psds_mean, x=freqs, axis=-1)
    
    # Compute power per band per channel
    band_powers = {}
    for band_name, (fmin, fmax) in BANDS.items():
        idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
        bp = simpson(psds_mean[:, idx_band], x=freqs[idx_band], axis=-1)
        band_powers[band_name] = bp
        
        # Store absolute and relative power per channel
        for i, ch in enumerate(ch_names):
            features[f"{ch}_{band_name}_abs"] = bp[i]
            features[f"{ch}_{band_name}_rel"] = bp[i] / (total_power_per_ch[i] + 1e-12)
            
    # Calculate Theta / Beta Ratio (TBR) - Classical ADHD biomarker
    theta_power = band_powers["theta"]
    beta_power = band_powers["beta"]
    tbr_per_ch = theta_power / (beta_power + 1e-12)
    
    for i, ch in enumerate(ch_names):
        features[f"{ch}_TBR"] = tbr_per_ch[i]
        
    # Canonical Frontal / Central TBR summary metrics (Fz, Cz)
    if "Fz" in ch_names:
        idx_fz = ch_names.index("Fz")
        features["TBR_Fz"] = tbr_per_ch[idx_fz]
    if "Cz" in ch_names:
        idx_cz = ch_names.index("Cz")
        features["TBR_Cz"] = tbr_per_ch[idx_cz]
    if "F3" in ch_names and "F4" in ch_names:
        idx_f3 = ch_names.index("F3")
        idx_f4 = ch_names.index("F4")
        # Frontal Alpha Asymmetry (FAA): ln(F4_alpha) - ln(F3_alpha)
        f4_alpha = band_powers["alpha"][idx_f4]
        f3_alpha = band_powers["alpha"][idx_f3]
        features["FAA_F4_F3"] = np.log(f4_alpha + 1e-12) - np.log(f3_alpha + 1e-12)
        
    return features

def export_epochs_tensor(epochs):
    """
    Exports clean epoch tensor as float32 numpy array:
    Shape: (n_epochs, n_channels, n_timepoints)
    """
    data = epochs.get_data(copy=True).astype(np.float32)
    return data
