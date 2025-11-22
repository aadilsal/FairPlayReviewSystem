from pathlib import Path
from train_ball_detector import BallDetectorTrainer


def find_latest_weights(repo_root: Path) -> Path | None:
    weights_dir = repo_root / 'weights'
    project_dir = repo_root / 'runs'

    candidates = []
    if weights_dir.exists():
        candidates.extend(list(weights_dir.rglob('*.pt')))

    if project_dir.exists():
        candidates.extend(list(project_dir.rglob('*.pt')))

    if not candidates:
        return None

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def pick_test_image(data_yaml_path: Path) -> Path | None:
    # Data yaml points to relative test images dir like '../test/images'
    import yaml
    try:
        d = yaml.safe_load(data_yaml_path.read_text())
        test_rel = d.get('test')
        if not test_rel:
            return None
        test_dir = (data_yaml_path.parent / test_rel).resolve()
        imgs = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
        return imgs[0] if imgs else None
    except Exception:
        return None


if __name__ == '__main__':
    repo_root = Path(__file__).resolve().parent
    data_yaml = repo_root / 'white ball detection.v1i.yolov8' / 'data.yaml'
    print(f"Using data yaml: {data_yaml}")

    trainer = BallDetectorTrainer(data_yaml=str(data_yaml))

    weights = find_latest_weights(repo_root)
    if weights is None:
        print("✗ No weight files found in `weights/` or `runs/`. Please train first or provide a path.")
    else:
        test_image = pick_test_image(data_yaml)
        if test_image is None:
            print("No test images found in dataset test split. You can pass a specific image to test.")
            trainer.test_inference(weights_path=str(weights), test_image=None)
        else:
            print(f"Running inference on: {test_image} using weights: {weights}")
            trainer.test_inference(weights_path=str(weights), test_image=str(test_image))
