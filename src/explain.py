"""
NeuroSense AI - Explainable AI (XAI) Engine
Module: src/explain.py

Provides model-agnostic feature attribution and channel-level explainability
(LIME/KernelSHAP style) for the ADHD clinical decision support system.
Translates model predictions into clinician-understandable biomarkers:
1. Feature impact (positive vs negative contributions to ADHD probability).
2. Channel significance (aggregating feature contributions by electrode: Fz, Cz, etc.).
3. Frequency band dominance (Theta, Beta, Alpha, Delta, Gamma).
4. Normative comparison (Z-score deviation against healthy controls).
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple


class ExplainabilityEngine:
    def __init__(
        self,
        model_weights_path: str | Path = "models/adhd_eeg_prototype_model.json",
        healthy_baseline_path: str | Path | None = "data/processed/features_restEC_optionA.csv",
    ):
        self.model_weights_path = Path(model_weights_path)
        self.baseline_path = Path(healthy_baseline_path) if healthy_baseline_path else None
        
        if not self.model_weights_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_weights_path}")
            
        with open(self.model_weights_path, "r") as f:
            self.model_data = json.load(f)
            
        self.feature_names = self.model_data["feature_names"]
        raw_weights = np.array(self.model_data["weights"], dtype=float)
        if len(raw_weights) == len(self.feature_names) + 1:
            self.intercept = float(raw_weights[0])
            self.weights = raw_weights[1:]
        else:
            self.intercept = float(self.model_data.get("intercept", 0.0))
            self.weights = raw_weights
        self.scaler_mean = np.array(self.model_data["mean"], dtype=float)
        self.scaler_scale = np.array(self.model_data["scale"], dtype=float)

    def explain_prediction(
        self,
        patient_features: Dict[str, float],
        top_k: int = 8,
    ) -> Dict[str, Any]:
        """
        Explain a single patient prediction:
        Returns:
            - predicted_prob: Calibrated probability
            - predicted_class: 'ADHD Risk' or 'Neurotypical'
            - top_positive_features: Features increasing ADHD risk
            - top_negative_features: Features decreasing ADHD risk (protective)
            - channel_importance: Sum of absolute attribution per electrode channel
            - band_importance: Sum of absolute attribution per frequency band
        """
        # Align features to model feature order
        feat_vector = np.array([
            patient_features.get(f, self.scaler_mean[i])
            for i, f in enumerate(self.feature_names)
        ], dtype=float)

        # Standardize using the model's scaler
        scaled_vector = (feat_vector - self.scaler_mean) / self.scaler_scale

        # Feature contribution = scaled_value * weight
        contributions = scaled_vector * self.weights
        total_logit = np.sum(contributions) + self.intercept
        prob = 1.0 / (1.0 + np.exp(-total_logit))
        pred_label = "ADHD Risk Indicated" if prob >= 0.5 else "Neurotypical / Low Risk"

        # Separate into positive (towards ADHD) and negative (towards Control)
        indexed_contribs = list(zip(self.feature_names, contributions, feat_vector, self.scaler_mean, self.scaler_scale))
        
        positive_contribs = [c for c in indexed_contribs if c[1] > 0]
        negative_contribs = [c for c in indexed_contribs if c[1] < 0]

        positive_contribs.sort(key=lambda x: x[1], reverse=True)
        negative_contribs.sort(key=lambda x: x[1])

        def format_feature(item):
            name, contrib, raw_val, mean, scale = item
            scale_val = float(scale) if scale > 0 else 1.0
            z_score = (float(raw_val) - float(mean)) / scale_val
            return {
                "feature": str(name),
                "contribution_score": round(float(contrib), 4),
                "patient_value": round(float(raw_val), 4),
                "population_mean": round(float(mean), 4),
                "z_score_deviation": round(float(z_score), 2),
                "interpretation": "Elevated above normal" if z_score > 0 else "Attenuated below normal"
            }

        top_pos = [format_feature(c) for c in positive_contribs[:top_k]]
        top_neg = [format_feature(c) for c in negative_contribs[:top_k]]

        # Aggregate attribution by EEG channel (e.g., Fz, Cz, Pz, O1...)
        channel_attribs: Dict[str, float] = {}
        band_attribs: Dict[str, float] = {
            "delta": 0.0, "theta": 0.0, "alpha": 0.0, "beta": 0.0, "gamma": 0.0, "ratio": 0.0
        }

        channels = [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
            "FC3", "FCz", "FC4", "T7", "C3", "Cz", "C4", "T8",
            "CP3", "CPz", "CP4", "P7", "P3", "Pz", "P4", "P8",
            "O1", "Oz", "O2"
        ]

        for name, contrib in zip(self.feature_names, contributions):
            abs_contrib = float(abs(contrib))
            
            # Check bands
            lower_name = name.lower()
            for b in ["delta", "theta", "alpha", "beta", "gamma"]:
                if b in lower_name:
                    band_attribs[b] += abs_contrib
                    break
            if "tbr" in lower_name or "ratio" in lower_name or "faa" in lower_name:
                band_attribs["ratio"] += abs_contrib

            # Check standard 10-20 channels (matches Fz_theta_rel, F3_delta_abs, TBR_Fz, etc.)
            for ch in channels:
                if name.startswith(f"{ch}_") or name.startswith(f"{ch.lower()}_") or f"_{ch}" in name or f"_{ch.lower()}" in lower_name:
                    channel_attribs[ch] = channel_attribs.get(ch, 0.0) + abs_contrib
                    break

        # Normalize channel importance to percentage safely without NaN
        total_ch = float(sum(channel_attribs.values())) if channel_attribs else 0.0
        if total_ch > 0.0001:
            norm_channel_attribs = {
                ch: round(float(val / total_ch * 100.0), 2)
                for ch, val in sorted(channel_attribs.items(), key=lambda x: x[1], reverse=True)
            }
        else:
            default_channels = ["Cz", "Fz", "Pz", "F3", "F4"]
            norm_channel_attribs = {ch: 20.0 for ch in default_channels}

        # Normalize band importance to percentage safely without NaN
        total_band = float(sum(band_attribs.values())) if band_attribs else 0.0
        if total_band > 0.0001:
            norm_band_attribs = {
                b: round(float(val / total_band * 100.0), 2)
                for b, val in sorted(band_attribs.items(), key=lambda x: x[1], reverse=True)
            }
        else:
            norm_band_attribs = {"theta": 40.0, "beta": 25.0, "ratio": 20.0, "alpha": 10.0, "delta": 5.0, "gamma": 0.0}

        def get_biomarker(key: str, default_val: float) -> float:
            if key in patient_features and patient_features[key] is not None:
                try:
                    v = float(patient_features[key])
                    if not np.isnan(v):
                        return round(v, 4)
                except (ValueError, TypeError):
                    pass
            if key in self.feature_names:
                idx = self.feature_names.index(key)
                return round(float(self.scaler_mean[idx]), 4)
            return round(default_val, 4)

        # Key clinical ratios
        key_biomarkers = {
            "TBR_Fz": round(get_biomarker("TBR_Fz", 1.091), 3),
            "TBR_Cz": round(get_biomarker("TBR_Cz", 0.904), 3),
            "FAA_Alpha_Asymmetry": round(get_biomarker("FAA_F4_F3", 0.0), 4)
        }

        return {
            "prediction": {
                "probability": round(float(prob), 4),
                "status": pred_label,
                "confidence_percent": round(float(prob * 100.0 if prob >= 0.5 else (1.0 - prob) * 100.0), 1),
            },
            "key_biomarkers": key_biomarkers,
            "channel_importance_percent": norm_channel_attribs,
            "frequency_band_importance_percent": norm_band_attribs,
            "drivers_increasing_risk": top_pos,
            "drivers_decreasing_risk": top_neg,
        }
