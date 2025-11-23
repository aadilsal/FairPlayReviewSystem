import os
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
ALLOWED_EXT = {"mp4", "avi", "mov", "mkv"}
MAX_MB = int(os.getenv("MAX_VIDEO_SIZE", 500))

# Helper functions for console logging
def log(msg, level="INFO"):
    """Print formatted log message to console"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    icons = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️", "DEBUG": "🔍"}
    icon = icons.get(level, "ℹ️")
    print(f"[{timestamp}] {icon} FRONTEND - {msg}")


def valid_filename(filename: str) -> bool:
    ext = filename.split('.')[-1].lower()
    return ext in ALLOWED_EXT


def main():
    log("="*60)
    log("Streamlit Frontend Application Starting")
    log(f"Backend URL: {BACKEND_URL}")
    log(f"Max file size: {MAX_MB} MB")
    log("="*60)
    
    st.set_page_config(page_title="Video Inference", layout="centered")
    st.title("🎬 Video Inference — Upload and Run")

    st.info(f"Upload a video (.mp4, .avi, .mov, .mkv). Max size: {MAX_MB} MB")

    uploaded = st.file_uploader("Choose a video file", type=list(ALLOWED_EXT))

    if uploaded is not None:
        size_mb = len(uploaded.getbuffer()) / (1024 * 1024)
        log(f"File selected: {uploaded.name} ({size_mb:.2f} MB)")
        st.write(f"📁 File: **{uploaded.name}** — {size_mb:.2f} MB")

        if not valid_filename(uploaded.name):
            log(f"Invalid file format: {uploaded.name}", "ERROR")
            st.error("❌ Unsupported file format.")
            return

        if size_mb > MAX_MB:
            log(f"File size warning: {size_mb:.2f} MB > {MAX_MB} MB", "WARNING")
            st.warning(f"⚠️ File size exceeds {MAX_MB} MB. Upload may fail or be slow.")

        if st.button("🚀 Start Processing"):
            log("\n" + "="*60)
            log("PROCESSING REQUEST STARTED", "SUCCESS")
            log("="*60)
            log(f"Uploading {uploaded.name} to backend...")
            
            with st.spinner("⏳ Uploading and processing..."):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue())}
                    log(f"Sending POST request to {BACKEND_URL}/api/predict")
                    
                    resp = requests.post(f"{BACKEND_URL}/api/predict", files=files, timeout=600)
                    log(f"Response status code: {resp.status_code}")
                    resp.raise_for_status()
                    
                    log("Upload successful, parsing response...", "SUCCESS")
                except requests.exceptions.RequestException as e:
                    log(f"Request failed: {e}", "ERROR")
                    st.error(f"❌ Upload failed: {e}")
                    return

                data = resp.json()
                log(f"Response data: status={data.get('status')}")
                
                if data.get("status") == "error":
                    error_msg = data.get("message", "Processing error")
                    log(f"Backend returned error: {error_msg}", "ERROR")
                    st.error(f"❌ {error_msg}")
                    return

                log("Processing completed successfully!", "SUCCESS")
                log(f"Run ID: {data.get('run_id')}")
                log(f"Inference time: {data.get('inference_time')}s")
                log(f"Frames processed: {data.get('frames_processed')}")
                log(f"Total detections: {data.get('total_detections')}")
                log("="*60 + "\n")
                
                st.success("✅ Processing finished!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ Inference Time", f"{data.get('inference_time', 0):.2f}s")
                with col2:
                    st.metric("🎞️ Frames", data.get('frames_processed', 0))
                with col3:
                    st.metric("🔍 Detections", data.get('total_detections', 0))
                
                st.write("**📝 MLflow Run ID:**", data.get("run_id"))

                preds = data.get("predictions")
                if preds is not None:
                    st.subheader("🎯 Detection Results")
                    st.json(preds)
                    log(f"Displayed {len(preds)} detection results")

                if data.get("result_file"):
                    res_path = data['result_file']
                    st.info(f"💾 Results saved: `{res_path}`")
                    log(f"Results file: {res_path}")


if __name__ == "__main__":
    main()
