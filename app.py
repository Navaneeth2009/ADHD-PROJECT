"""
NeuroSense AI - Clinician Web Dashboard Server
Module: app.py

A lightweight, responsive web server providing clinicians with a clean interface to:
1. Upload/Drag-and-Drop EEG files (.bdf, .csv, .json).
2. Select psychiatric disorder module (ADHD, Depression, Schizophrenia, etc.).
3. Choose inference engine (Spectral Ridge Biomarker Model or EEGNet Neural Network).
4. View real-time predictions, calibrated confidence meters, and Explainable AI (XAI) topoplot/channel bars.
5. Export clinical decision support reports with one click.
"""

from __future__ import annotations

import os
import sys
import json
import uuid
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Set workspace root
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.explain import ExplainabilityEngine
from src.report import generate_clinical_report

MODEL_FILE = BASE_DIR / "models" / "adhd_eeg_prototype_model.json"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

explain_engine = ExplainabilityEngine(model_weights_path=MODEL_FILE)

# Pre-load TDBRAIN cohort dataset registry for instant patient lookup
import re
import pandas as pd

DATASET_CSV = BASE_DIR / "data" / "processed" / "features_restEC_optionA.csv"
DATASET_REGISTRY = {}
if DATASET_CSV.exists():
    try:
        _df = pd.read_csv(DATASET_CSV)
        for _, _row in _df.iterrows():
            DATASET_REGISTRY[str(_row["subject_id"])] = _row.to_dict()
        print(f">> Loaded {len(DATASET_REGISTRY)} TDBRAIN cohort subjects into memory registry.")
    except Exception as _e:
        print("Warning loading dataset registry:", _e)

# Also index all pre-extracted patient JSON files in sample_patients/
SAMPLES_DIR = BASE_DIR / "sample_patients"
if SAMPLES_DIR.exists():
    for _p in SAMPLES_DIR.glob("*.json"):
        try:
            with open(_p, "r", encoding="utf-8") as _f:
                _data = json.load(_f)
                _sid = str(_data.get("subject_id", ""))
                if not _sid:
                    _m = re.search(r"(sub-\d+)", _p.stem)
                    if _m:
                        _sid = _m.group(1)
                if _sid:
                    DATASET_REGISTRY[_sid] = _data
                    # Also index with prefix if needed
                    _stem = _p.stem
                    DATASET_REGISTRY[_stem] = _data
        except Exception as _e:
            pass
    print(f">> Registry now indexed {len(DATASET_REGISTRY)} total subject feature sets.")

DASHBOARD_HTML_FILE = BASE_DIR / "neuro_dashboard.html"

def get_dashboard_html() -> bytes:
    if DASHBOARD_HTML_FILE.exists():
        return DASHBOARD_HTML_FILE.read_bytes()
    return b"<h1>NeuroSense AI: Dashboard HTML file not found</h1>"


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        if parsed.path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(get_dashboard_html())
            return

        elif parsed.path == "/report":
            # Generate or serve existing report
            report_file = REPORTS_DIR / "test_cli_report.html"
            if not report_file.exists():
                sample_file = BASE_DIR / "sample_input.json"
                if sample_file.exists():
                    feats = json.load(open(sample_file))
                    exp = explain_engine.explain_prediction(feats)
                    report_file = generate_clinical_report(exp, output_path=report_file)

            if report_file.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(report_file.read_bytes())
                return

        elif parsed.path == "/api/sample":
            sample_file = BASE_DIR / "sample_input.json"
            if sample_file.exists():
                with open(sample_file, "r") as f:
                    data = json.load(f)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode("utf-8"))
                return

        elif parsed.path == "/api/patient":
            query = urllib.parse.parse_qs(parsed.query)
            pid = query.get("id", [None])[0]
            if pid:
                sub_match = re.search(r"(sub-\d+)", str(pid))
                sub_key = sub_match.group(1) if sub_match else str(pid)
                if sub_key in DATASET_REGISTRY:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(DATASET_REGISTRY[sub_key]).encode("utf-8"))
                    return
                for cand in [sub_key, f"ADHD_{sub_key}", f"HEALTHY_{sub_key}"]:
                    cand_path = BASE_DIR / "sample_patients" / f"{cand}.json"
                    if cand_path.exists():
                        try:
                            data = json.loads(cand_path.read_text(encoding="utf-8"))
                            DATASET_REGISTRY[sub_key] = data
                            self.send_response(200)
                            self.send_header("Content-Type", "application/json")
                            self.end_headers()
                            self.wfile.write(json.dumps(data).encode("utf-8"))
                            return
                        except Exception:
                            pass
            self.send_error(404, f"Patient {pid} not found in database registry")
            return

        self.send_error(404, "Page Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                payload = json.loads(body.decode("utf-8"))
                features = payload.get("features", {})
                subject_id = payload.get("subject_id", "PATIENT-DEMO")
                
                # Check if features are empty or hardware metadata without spectral biomarkers
                has_model_feats = any(k in features for k in ["TBR_Fz", "Cz_theta_rel", "Fz_theta_rel", "Fp1_delta_abs"])
                sub_match = re.search(r"(sub-\d+)", str(subject_id))
                sub_id = sub_match.group(1) if sub_match else None
                
                # Auto-resolve from cohort registry if incoming data was raw metadata
                if not has_model_feats and sub_id and sub_id in DATASET_REGISTRY:
                    features = DATASET_REGISTRY[sub_id]
                    subject_id = sub_id

                explanation = explain_engine.explain_prediction(features)
                
                # Also auto-update the HTML report for this patient
                report_path = REPORTS_DIR / "clinical_report.html"
                generate_clinical_report(explanation, patient_id=subject_id, output_path=report_path)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(explanation).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        elif parsed.path == "/api/upload_bdf":
            content_length = int(self.headers.get("Content-Length", 0))
            raw_data = self.rfile.read(content_length)
            subject_id = urllib.parse.unquote(self.headers.get("X-Subject-ID", "UPLOADED_PATIENT"))
            
            sub_match = re.search(r"(sub-\d+)", str(subject_id))
            sub_key = sub_match.group(1) if sub_match else None

            # If this is a known subject from our TDBRAIN adult cohort, use preprocessed features directly
            if sub_key and sub_key in DATASET_REGISTRY:
                reg_features = DATASET_REGISTRY[sub_key]
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "subject_id": sub_key,
                    "retained_epochs": int(reg_features.get("n_epochs", 117)),
                    "features": reg_features
                }).encode("utf-8"))
                return
            
            try:
                temp_upload_path = BASE_DIR / "scratch" / f"temp_{uuid.uuid4().hex[:8]}.bdf"
                temp_upload_path.parent.mkdir(parents=True, exist_ok=True)
                temp_upload_path.write_bytes(raw_data)
                
                from src.preprocess import preprocess_eeg_subject
                from src.features import compute_band_powers
                
                epochs, stats = preprocess_eeg_subject(temp_upload_path, apply_ica=True)
                features = compute_band_powers(epochs)
                
                # Cleanup temp file
                if temp_upload_path.exists():
                    temp_upload_path.unlink()

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "subject_id": subject_id,
                    "retained_epochs": stats.get("n_epochs_retained", 60),
                    "features": features
                }).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Signal processing failed: {str(e)}"}).encode("utf-8"))
            return

        self.send_error(404, "Endpoint Not Found")


def run_server(port: int = 8055):
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])

    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    url = f"http://localhost:{port}"
    print(f"\n=======================================================")
    print(f">> NeuroSense AI Clinician Dashboard Live at: {url}")
    print(f">> (Keep this terminal window open while using the dashboard)")
    print(f"=======================================================\n")

    import webbrowser
    import threading
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    server.serve_forever()


if __name__ == "__main__":
    run_server()
