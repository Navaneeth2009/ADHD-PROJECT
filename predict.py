import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Project root
BASE_DIR = Path("d:/ADHD_Project")
sys.path.append(str(BASE_DIR))

MODEL_FILE = BASE_DIR / "models" / "adhd_eeg_prototype_model.json"
DATA_FEATURES = BASE_DIR / "data" / "processed" / "features_restEC_optionA.csv"

def load_trained_model(model_path=MODEL_FILE):
    """Loads the trained model weights, scales, and feature list from JSON."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    with open(model_path, "r") as f:
        data = json.load(f)
    return data

def predict_from_features(feature_dict, model_data):
    """
    Takes a dictionary of EEG features and computes the ADHD classification.
    """
    feature_names = model_data["feature_names"]
    mean = np.array(model_data["mean"])
    scale = np.array(model_data["scale"])
    weights = np.array(model_data["weights"])

    # Extract feature values in exact model order
    vector = np.array([feature_dict.get(feat, 0.0) for feat in feature_names], dtype=float)
    
    # Standardize
    vector_s = (vector - mean) / scale
    vector_bias = np.insert(vector_s, 0, 1.0)

    # Compute prediction score and probability
    score = np.dot(vector_bias, weights)
    prob_adhd = 1.0 / (1.0 + np.exp(-np.clip(score, -30, 30)))
    prediction = "ADHD" if prob_adhd >= 0.5 else "HEALTHY"
    confidence = prob_adhd if prob_adhd >= 0.5 else (1.0 - prob_adhd)

    return {
        "prediction": prediction,
        "adhd_probability": round(float(prob_adhd), 4),
        "confidence": round(float(confidence), 4),
        "key_biomarkers": {
            "TBR_Fz": round(float(feature_dict.get("TBR_Fz", 0.0)), 3),
            "TBR_Cz": round(float(feature_dict.get("TBR_Cz", 0.0)), 3),
            "F8_theta_rel": round(float(feature_dict.get("F8_theta_rel", 0.0)), 3),
            "Cz_theta_rel": round(float(feature_dict.get("Cz_theta_rel", 0.0)), 3)
        }
    }

def predict_from_bdf(bdf_file_path, model_data):
    """
    Processes a raw .bdf recording file directly through the MNE pipeline
    and outputs the diagnosis.
    """
    # Import pipeline preprocessing modules
    from src.preprocess import preprocess_eeg_subject
    from src.features import compute_band_powers

    print(f"Preprocessing raw EEG file: {bdf_file_path}...")
    epochs, stats = preprocess_eeg_subject(bdf_file_path, apply_ica=True)
    features = compute_band_powers(epochs)
    result = predict_from_features(features, model_data)
    result["retained_epochs"] = stats["n_epochs_retained"]
    return result

def predict_by_subject_id(subject_id, model_data):
    """
    Finds an existing processed subject or raw file and returns prediction.
    """
    if DATA_FEATURES.exists():
        df = pd.read_csv(DATA_FEATURES)
        match = df[df["subject_id"] == subject_id]
        if len(match) > 0:
            feat_dict = match.iloc[0].to_dict()
            actual = feat_dict.get("indication", "Unknown")
            res = predict_from_features(feat_dict, model_data)
            res["subject_id"] = subject_id
            res["ground_truth"] = actual
            return res

    # If not in CSV, check raw directory
    from src.config import DATASET_DIR
    bdf_file = DATASET_DIR / subject_id / "ses-1" / "eeg" / f"{subject_id}_ses-1_task-restEC_eeg.bdf"
    if bdf_file.exists():
        res = predict_from_bdf(bdf_file, model_data)
        res["subject_id"] = subject_id
        return res

    raise ValueError(f"Subject ID {subject_id} not found in processed data or raw dataset.")

def interactive_cli():
    parser = argparse.ArgumentParser(description="NeuroSense AI: Predict ADHD from EEG Input with Explainable AI")
    parser.add_argument("--bdf", type=str, default=None, help="Path to raw .bdf EEG recording file")
    parser.add_argument("--csv", type=str, default=None, help="Path to a CSV file containing EEG features or rows")
    parser.add_argument("--json", type=str, default=None, help="JSON string or file path with EEG feature key-value pairs")
    parser.add_argument("--subject-id", type=str, default=None, help="Subject ID from dataset (e.g. sub-88005441)")
    parser.add_argument("--model", type=str, default="ridge", choices=["ridge", "eegnet"], help="Inference model: 'ridge' (default, fast spectral) or 'eegnet' (neural network)")
    parser.add_argument("--explain", action="store_true", help="Generate Explainable AI (XAI) feature and channel attribution")
    parser.add_argument("--export-report", type=str, default=None, help="Export clinical HTML decision support report to specified file path")
    args = parser.parse_args()

    model_data = load_trained_model()

    # Determine input features
    features_dict = None
    target_subject_id = args.subject_id or "DEMO-PATIENT"

    if args.bdf:
        if args.model == "eegnet":
            from src.eegnet_model import PureEEGNet
            from src.preprocess import preprocess_eeg_subject, export_epochs_tensor
            print(f"Running EEGNet Neural Network on raw BDF: {args.bdf}...")
            epochs, stats = preprocess_eeg_subject(args.bdf, apply_ica=True)
            tensor = export_epochs_tensor(epochs)
            net = PureEEGNet(n_channels=26, n_samples=1001)
            res = net.predict_subject(tensor)
            print("\n" + "="*50)
            print("EEGNET NEURAL NETWORK PREDICTION RESULT:")
            print("="*50)
            print(json.dumps(res, indent=2))
            return
            
        from src.preprocess import preprocess_eeg_subject
        from src.features import compute_band_powers
        print(f"Preprocessing raw EEG file: {args.bdf}...")
        epochs, stats = preprocess_eeg_subject(args.bdf, apply_ica=True)
        features_dict = compute_band_powers(epochs)
        target_subject_id = Path(args.bdf).stem.split("_")[0]
        res = predict_from_features(features_dict, model_data)
        res["retained_epochs"] = stats["n_epochs_retained"]
        print("\n" + "="*50)
        print("PREDICTION FROM RAW EEG (.bdf):")
        print("="*50)
        print(json.dumps(res, indent=2))

    elif args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {args.csv}")
        df = pd.read_csv(csv_path)
        print(f"\nProcessing {len(df)} row(s) from {args.csv}...")
        row = df.iloc[0].to_dict()
        features_dict = row
        target_subject_id = row.get("subject_id", "CSV_PATIENT")
        res = predict_from_features(features_dict, model_data)
        res["subject_id"] = target_subject_id
        print("\n" + "="*50)
        print("PREDICTION RESULT:")
        print("="*50)
        print(json.dumps(res, indent=2))

    elif args.json:
        json_path = Path(args.json)
        if json_path.exists():
            with open(json_path, "r") as f:
                features_dict = json.load(f)
        else:
            features_dict = json.loads(args.json)
        res = predict_from_features(features_dict, model_data)
        print("\n" + "="*50)
        print("PREDICTION FROM JSON FEATURE INPUT:")
        print("="*50)
        print(json.dumps(res, indent=2))

    elif args.subject_id:
        res = predict_by_subject_id(args.subject_id, model_data)
        target_subject_id = args.subject_id
        if DATA_FEATURES.exists():
            df = pd.read_csv(DATA_FEATURES)
            match = df[df["subject_id"] == args.subject_id]
            if len(match) > 0:
                features_dict = match.iloc[0].to_dict()
        print("\n" + "="*50)
        print(f"PREDICTION FOR {args.subject_id}:")
        print("="*50)
        print(json.dumps(res, indent=2))

    else:
        # Default Demo
        print("\n" + "="*55)
        print("DEMO: PREDICTING FROM EEG (1 ADHD vs 1 HEALTHY)")
        print("="*55)
        test_subjects = ["sub-88062189", "sub-87957837"]
        for sid in test_subjects:
            r = predict_by_subject_id(sid, model_data)
            print(f"\n--- Patient: {sid} ---")
            print(f"  Actual Diagnosis : {r.get('ground_truth', 'Unknown')}")
            print(f"  Model Prediction : {r['prediction']} (Confidence: {r['confidence']*100:.1f}%)")
            print(f"  ADHD Probability : {r['adhd_probability']:.4f}")
            print(f"  Biomarkers       : TBR Fz = {r['key_biomarkers']['TBR_Fz']}, Cz = {r['key_biomarkers']['TBR_Cz']}")
        return

    # Handle Explainability and Report Export
    if (args.explain or args.export_report) and features_dict is not None:
        from src.explain import ExplainabilityEngine
        engine = ExplainabilityEngine()
        explanation = engine.explain_prediction(features_dict)

        if args.explain:
            print("\n" + "="*50)
            print("EXPLAINABLE AI (XAI) ATTRIBUTION SUMMARY:")
            print("="*50)
            print(f"Decision Status      : {explanation['prediction']['status']}")
            print(f"Calibrated Confidence: {explanation['prediction']['confidence_percent']}%")
            print("\nTop Contributing Scalp Electrodes:")
            for ch, pct in list(explanation['channel_importance_percent'].items())[:5]:
                print(f"  • Electrode {ch:<4} : {pct}% of decision variance")
            print("\nDominant Frequency Bands:")
            for b, pct in list(explanation['frequency_band_importance_percent'].items())[:3]:
                print(f"  • {b.capitalize():<6} Power : {pct}%")

        if args.export_report:
            from src.report import generate_clinical_report
            rep_path = generate_clinical_report(
                explanation,
                patient_id=target_subject_id,
                output_path=args.export_report
            )
            print("\n" + "="*50)
            print(f"CLINICAL REPORT EXPORTED: file:///{Path(rep_path).resolve()}")
            print("="*50)

if __name__ == "__main__":
    interactive_cli()
