"""Small helpers to interact with Roboflow (download dataset or hosted model).

This script uses the `roboflow` python package. Set `ROBOFLOW_API_KEY` in
your environment (or .env) before running.

Examples:
  python -m mlops.roboflow_helper download_dataset <workspace/project> <version>
  python -m mlops.roboflow_helper download_model <workspace/project> <version> --format yolov8
"""
import sys
import argparse
import os
from pathlib import Path

try:
    from roboflow import Roboflow
except Exception:
    Roboflow = None


def download_dataset(project, version, out_dir="./roboflow_data"):
    if Roboflow is None:
        raise RuntimeError("roboflow package not installed. pip install roboflow")
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    ws, name = project.split("/", 1) if "/" in project else (None, project)
    if ws:
        p = rf.workspace(ws).project(name)
    else:
        p = rf.project(name)
    v = p.version(int(version))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset to {out_dir}...")
    v.download("yolov8", out_dir)


def download_model(project, version, fmt="yolov8", out_dir="./roboflow_models"):
    if Roboflow is None:
        raise RuntimeError("roboflow package not installed. pip install roboflow")
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    ws, name = project.split("/", 1) if "/" in project else (None, project)
    if ws:
        p = rf.workspace(ws).project(name)
    else:
        p = rf.project(name)
    v = p.version(int(version))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Downloading hosted model (format={fmt}) to {out_dir}...")
    v.model.download(fmt, out_dir)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    d_ds = sub.add_parser("download_dataset")
    d_ds.add_argument("project")
    d_ds.add_argument("version")
    d_ds.add_argument("--out", default="./roboflow_data")

    d_m = sub.add_parser("download_model")
    d_m.add_argument("project")
    d_m.add_argument("version")
    d_m.add_argument("--format", default="yolov8")
    d_m.add_argument("--out", default="./roboflow_models")

    args = parser.parse_args()
    if args.cmd == "download_dataset":
        download_dataset(args.project, args.version, args.out)
    elif args.cmd == "download_model":
        download_model(args.project, args.version, args.format, args.out)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
