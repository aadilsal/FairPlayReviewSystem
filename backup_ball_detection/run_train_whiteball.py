from pathlib import Path
from train_ball_detector import BallDetectorTrainer

if __name__ == '__main__':
    repo_root = Path(__file__).resolve().parent
    data_yaml = repo_root / 'white ball detection.v1i.yolov8' / 'data.yaml'
    print(f"Using data yaml: {data_yaml}")
    trainer = BallDetectorTrainer(data_yaml=str(data_yaml))
    # recommend model size 's' for balance
    trainer.train_from_scratch(model_size='s', delete_old_weights=False)
