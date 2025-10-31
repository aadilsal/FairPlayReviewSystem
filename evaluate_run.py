"""Evaluate a trained YOLO model and save metrics.

Usage:
    python evaluate_run.py --run-name yolov8s-cricket-ball --data cricket_ball_data/data.yaml --out metrics/yolov8s-cricket-ball.yaml

The script looks for the weights at runs/train/<run_name>/weights/best.pt by default.
It loads the model with ultralytics.YOLO and runs validation, then writes a YAML/JSON
metrics file suitable for DVC metrics collection.
"""
import argparse
import json
from pathlib import Path
import yaml

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run-name', required=False, help='Name of the training run (runs/train/<name>)')
    p.add_argument('--weights', required=False, help='Path to weights file (overrides run-name)')
    p.add_argument('--data', required=True, help='Path to dataset yaml')
    p.add_argument('--out', required=True, help='Output metrics file (yaml or json)')
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve weights path
    if args.weights:
        weights_path = Path(args.weights)
    elif args.run_name:
        weights_path = Path('runs') / 'train' / args.run_name / 'weights' / 'best.pt'
    else:
        raise SystemExit('Either --weights or --run-name must be provided')

    if not weights_path.exists():
        raise SystemExit(f'Weights not found: {weights_path}')

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise SystemExit(f'Unable to import ultralytics: {e}')

    model = YOLO(str(weights_path))

    print(f'[INFO] Running validation with {weights_path} on {args.data}')
    # val() prints and returns a dict of metrics
    results = model.val(data=args.data)

    # results may be a dict-like object. Convert to plain Python types
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        try:
            return float(obj)
        except Exception:
            return obj

    metrics = convert(results)

    # Save as YAML if extension is .yaml/.yml else JSON
    if out_path.suffix in ('.yaml', '.yml'):
        with open(out_path, 'w') as f:
            yaml.safe_dump(metrics, f)
    else:
        with open(out_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    # Also write a copy inside the run folder if run_name provided
    if args.run_name:
        run_metrics_path = Path('runs') / 'train' / args.run_name / 'metrics.yaml'
        with open(run_metrics_path, 'w') as f:
            yaml.safe_dump(metrics, f)

    print(f'[INFO] Metrics saved to {out_path}')


if __name__ == '__main__':
    main()
