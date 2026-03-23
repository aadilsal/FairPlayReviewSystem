import os
import uuid
import cv2
from fastapi import UploadFile, HTTPException

TMP_DIR = os.path.join("outputs", "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
MAX_VIDEO_SIZE_MB = 500

def save_upload_file(upload_file: UploadFile) -> str:
    ext = os.path.splitext(upload_file.filename or "")[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file format. Use mp4/mov/avi/mkv.")

    file_bytes = upload_file.file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    max_bytes = MAX_VIDEO_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_VIDEO_SIZE_MB}MB size limit.")

    filename = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(TMP_DIR, filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file_bytes)

    cap = cv2.VideoCapture(file_path)
    opened = cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    cap.release()
    if not opened or frame_count <= 0:
        delete_file(file_path)
        raise HTTPException(status_code=400, detail="Invalid or corrupted video file.")

    # Reset read pointer for safety in case caller reuses UploadFile object.
    try:
        upload_file.file.seek(0)
    except Exception:
        pass

    return file_path

def delete_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)
