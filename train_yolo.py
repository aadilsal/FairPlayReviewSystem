"""Simple YOLOv8 training script for transfer learning on your datasets.

Usage examples (PowerShell):
    python train_yolo.py -w yolov8s.pt -d cricket_ball_data/data.yaml -e 50 -n yolov8s-cricket-ball

This script uses the ultralytics.YOLO API which is already used elsewhere in
your project. It trains and saves weights into the `runs/train/...` folder.
"""
import argparse
from ultralytics import YOLO
import os
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 model (transfer learning)")
    p.add_argument('-w', '--weights', default='yolov8s.pt', help='Base weights to start from (e.g. yolov8s.pt)')
    p.add_argument('-d', '--data', default='cricket_ball_data/data.yaml', help='Dataset YAML path')
    p.add_argument('-e', '--epochs', type=int, default=50, help='Number of training epochs')
    p.add_argument('-b', '--batch', type=int, default=16, help='Batch size')
    p.add_argument('--imgsz', type=int, default=640, help='Image size')
    p.add_argument('-n', '--name', default='yolov8-transfer', help='Run name (output folder)')
    p.add_argument('--device', default='auto', help="Device to use: 'auto' (default), 'cpu', or CUDA ids like '0' or '0,1'")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset yaml not found: {args.data}")

    # Decide device automatically when requested
    device = args.device
    if device == 'auto':
        device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Training YOLO from {args.weights} on {args.data} using device={device}")

    model = YOLO(args.weights)

    try:
        # Basic training call - this uses Ultralytics' high-level API
        model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            name=args.name,
            device=device
        )
    except ValueError as e:
        print(f"[ERROR] Training failed due to device selection: {e}")
        if not torch.cuda.is_available():
            print("[HINT] No CUDA devices detected. Re-run with --device cpu or install a CUDA-enabled PyTorch build if you have a GPU.")
        raise

    print(f"[INFO] Training finished. Check runs/train/{args.name}/weights for best.pt")


if __name__ == '__main__':
    main()
