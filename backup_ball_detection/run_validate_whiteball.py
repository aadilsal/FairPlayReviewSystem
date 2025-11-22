from pathlib import Path
from train_ball_detector import BallDetectorTrainer


def find_latest_weights(repo_root: Path) -> Path | None:
    weights_dir = repo_root / 'weights'
    project_dir = repo_root / 'runs'

    candidates = []
    if weights_dir.exists():
        candidates.extend(list(weights_dir.rglob('*.pt')))

    # also check runs folder for .pt files
    if project_dir.exists():
        candidates.extend(list(project_dir.rglob('*.pt')))

    if not candidates:
        return None

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


if __name__ == '__main__':
    repo_root = Path(__file__).resolve().parent
    data_yaml = repo_root / 'white ball detection.v1i.yolov8' / 'data.yaml'
    print(f"Using data yaml: {data_yaml}")

    trainer = BallDetectorTrainer(data_yaml=str(data_yaml))

    weights = find_latest_weights(repo_root)
    if weights is None:
        print("✗ No weight files found in `weights/` or `runs/`. Please train first or provide a path.")
    else:
        print(f"Validating using weights: {weights}")
        trainer.validate_model(weights_path=str(weights))
