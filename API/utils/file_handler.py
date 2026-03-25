import os
import uuid
from fastapi import UploadFile
from typing import Tuple

TMP_DIR = os.path.join("outputs", "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

def save_upload_file(upload_file: UploadFile) -> str:
    ext = os.path.splitext(upload_file.filename)[1]
    filename = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(TMP_DIR, filename)
    with open(file_path, "wb") as buffer:
        buffer.write(upload_file.file.read())
    return file_path

def delete_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)
