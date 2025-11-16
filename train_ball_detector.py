from ultralytics import YOLO
import torch
from pathlib import Path
import shutil


class BallDetectorTrainer:
    def __init__(self, data_yaml="cricket_ball_data/data.yaml"):
        self.data_yaml = data_yaml
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        if self.device == 0:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"\n✓ Using GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
        else:
            print("\n⚠️  WARNING: GPU not available, using CPU (very slow!)")

        self.training_config = {
            'epochs': 200,
            'imgsz': 640,
            'batch': 8,
            'device': self.device,
            'workers': 4,
            'patience': 50,
            'save': True,
            'save_period': 10,
            'cache': True,
            'exist_ok': True,
            'pretrained': False,
            'verbose': True,
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.1,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.1,
            'copy_paste': 0.0,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'val': True,
            'plots': True,
            'save_json': True,
            'amp': True,
            'fraction': 0.9,
            'max_det': 300,
        }

    def clear_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✓ GPU cache cleared")

    def find_and_delete_weights(self, confirm=True):
        print(f"\n{'='*60}")
        print("SEARCHING FOR EXISTING WEIGHT FILES")
        print(f"{'='*60}\n")

        weight_extensions = ['.pt', '.pth', '.weights', '.onnx']
        weight_files = []
        search_paths = [Path('.'), Path('weights'), Path('models'), Path('runs')]

        for search_path in search_paths:
            if search_path.exists():
                for ext in weight_extensions:
                    weight_files.extend(list(search_path.rglob(f'*{ext}')))

        weight_files = list(set(weight_files))

        if not weight_files:
            print("✓ No existing weight files found\n")
            return

        print(f"Found {len(weight_files)} weight file(s):")
        total_size = 0
        for wf in weight_files:
            size_mb = wf.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"  - {wf} ({size_mb:.2f} MB)")

        print(f"\nTotal size: {total_size:.2f} MB")

        if confirm:
            response = input("\nDelete these files? (yes/no): ").strip().lower()
            if response != 'yes':
                print("Deletion cancelled")
                return

        deleted = 0
        for wf in weight_files:
            try:
                wf.unlink()
                deleted += 1
            except Exception as e:
                print(f"Failed to delete {wf}: {e}")

        print(f"\n✓ Deleted {deleted}/{len(weight_files)} weight files\n")

    def train_from_scratch(self, model_size='s', delete_old_weights=True):
        print(f"\n{'='*60}")
        print("TRAINING YOLO BALL DETECTOR FROM SCRATCH")
        print(f"{'='*60}\n")

        if delete_old_weights:
            self.find_and_delete_weights(confirm=True)

        self.clear_gpu_memory()

        if model_size == 'n':
            self.training_config['batch'] = 16
        elif model_size == 's':
            self.training_config['batch'] = 8
        elif model_size == 'm':
            self.training_config['batch'] = 4
        else:
            self.training_config['batch'] = 2

        print(f"Model: YOLOv8{model_size}")
        print(f"Batch size: {self.training_config['batch']}")
        print(f"Image size: {self.training_config['imgsz']}")
        print(f"Epochs: {self.training_config['epochs']}")
        if torch.cuda.is_available():
            print(f"Device: GPU ({torch.cuda.get_device_name(0)})")
        else:
            print("Device: CPU")
        print("Mixed Precision: Enabled\n")

        print("Loading YOLOv8 architecture...")
        model = YOLO(f'yolov8{model_size}.yaml')

        print("Starting training from scratch...\n")
        try:
            results = model.train(data=self.data_yaml, **self.training_config)

            print(f"\n{'='*60}")
            print("✓ TRAINING COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}\n")

            try:
                best_weights = Path('runs/detect/train/weights/best.pt')
                if best_weights.exists():
                    weights_dir = Path('weights')
                    weights_dir.mkdir(exist_ok=True)
                    final_weights = weights_dir / 'ball_detector_best.pt'
                    shutil.copy(best_weights, final_weights)
                    print(f"\n✓ Best weights saved to: {final_weights}")
            except Exception as e:
                print(f"Could not copy best weights: {e}")

            return results

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\n✗ GPU OUT OF MEMORY ERROR!")
                print("\nTry these solutions:")
                print("  1. Reduce batch size (currently: {})".format(self.training_config['batch']))
                print("  2. Use smaller model ('n' instead of 's')")
                print("  3. Reduce image size (try 512 or 416)")
                print("  4. Close other GPU applications")
                self.clear_gpu_memory()
            raise e

    def validate_model(self, weights_path='weights/ball_detector_best.pt'):
        print(f"\n{'='*60}")
        print("VALIDATING MODEL")
        print(f"{'='*60}\n")

        if not Path(weights_path).exists():
            print(f"✗ Weights file not found: {weights_path}")
            return None

        model = YOLO(weights_path)
        metrics = model.val(data=self.data_yaml)

        print("\nValidation Metrics:")
        try:
            print(f"  mAP@50: {metrics.box.map50:.4f}")
            print(f"  mAP@50-95: {metrics.box.map:.4f}")
            print(f"  Precision: {metrics.box.mp:.4f}")
            print(f"  Recall: {metrics.box.mr:.4f}")
        except Exception:
            print("Validation completed (metrics object format may vary).")

        return metrics

    def test_inference(self, weights_path='weights/ball_detector_best.pt', test_image=None):
        if not Path(weights_path).exists():
            print(f"✗ Weights file not found: {weights_path}")
            return

        model = YOLO(weights_path)

        if test_image is None:
            test_dir = Path('cricket_ball_data/test/images')
            if test_dir.exists():
                test_images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
                if test_images:
                    test_image = str(test_images[0])

        if test_image:
            print(f"\nRunning inference on: {test_image}")
            results = model.predict(test_image, save=True, conf=0.25)
            print("✓ Results saved to runs/detect/predict")


if __name__ == "__main__":
    import sys

    print("Checking GPU availability...")
    if not torch.cuda.is_available():
        print("\n⚠️  ERROR: GPU not detected!")
        print("Please install CUDA-enabled PyTorch first")
        sys.exit(1)

    trainer = BallDetectorTrainer()

    print("\nRecommendation: Use 's' (small) for best balance")
    model_size = input("Enter model size (n/s/m): ").strip().lower() or 's'

    results = trainer.train_from_scratch(model_size=model_size, delete_old_weights=True)

    print("\nValidating trained model...")
    trainer.validate_model()

    print("\nTesting inference...")
    trainer.test_inference()

    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Check training curves in: runs/detect/train/")
    print("  2. Review validation results")
    print("  3. Test on challenging images (white balls, blurry frames)")
    print("  4. Update detection pipeline to use: weights/ball_detector_best.pt")
