
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import shutil
from tqdm import tqdm


class BallDataAugmenter:
    """
    Augment cricket ball dataset optimized for batch processing
    Focus on white balls and blurry frames
    """
    def __init__(self, dataset_path="cricket_ball_data"):
        self.dataset_path = Path(dataset_path)
        self.train_images = self.dataset_path / "train" / "images"
        self.train_labels = self.dataset_path / "train" / "labels"

        # Augmentation pipeline optimized for ball detection
        self.augmentation_pipeline = A.Compose([
            A.OneOf([
                A.MotionBlur(blur_limit=(5, 20), p=1.0),
                A.GaussianBlur(blur_limit=(3, 13), p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
            ], p=0.6),

            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),

            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=40, val_shift_limit=30, p=1.0),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
            ], p=0.5),

            A.OneOf([
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
                A.RandomToneCurve(scale=0.3, p=1.0),
            ], p=0.4),

            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),

            A.OneOf([
                A.GaussNoise(var_limit=(15.0, 60.0), p=1.0),
                A.ISONoise(color_shift=(0.02, 0.08), intensity=(0.2, 0.6), p=1.0),
            ], p=0.35),

            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.25, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.15),

            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=0.2),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

    def parse_yolo_label(self, label_path):
        """Parse YOLO format label file"""
        if not label_path.exists():
            return [], []

        bboxes = []
        class_labels = []

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:]]
                    class_labels.append(class_id)
                    bboxes.append(bbox)

        return bboxes, class_labels

    def save_yolo_label(self, label_path, bboxes, class_labels):
        """Save YOLO format label file"""
        with open(label_path, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                line = f"{class_id} {' '.join([f'{x:.6f}' for x in bbox])}\n"
                f.write(line)

    def augment_dataset(self, multiplier=3, backup=True):
        """
        Augment training dataset
        multiplier: Number of augmented versions per original image
        backup: Create backup before augmentation
        """
        print(f"\n{'='*60}")
        print("BALL DATASET AUGMENTATION")
        print(f"{'='*60}\n")

        if backup:
            backup_path = self.dataset_path / "train_backup"
            if not backup_path.exists():
                print("Creating backup of original training data...")
                shutil.copytree(self.dataset_path / "train", backup_path)
                print(f"✓ Backup created at: {backup_path}\n")

        image_files = list(self.train_images.glob("*.jpg")) + list(self.train_images.glob("*.png")) + list(self.train_images.glob("*.jpeg"))
        original_count = len(image_files)
        print(f"Original training images: {original_count}")
        print(f"Generating {multiplier} augmented versions per image...")
        print(f"Total images after augmentation: {original_count * (multiplier + 1)}\n")

        augmented_count = 0

        for img_path in tqdm(image_files, desc="Augmenting images"):
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label_path = self.train_labels / f"{img_path.stem}.txt"
            bboxes, class_labels = self.parse_yolo_label(label_path)

            if not bboxes:
                continue

            for aug_idx in range(multiplier):
                try:
                    augmented = self.augmentation_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
                    aug_image = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_labels = augmented['class_labels']

                    if not aug_bboxes:
                        continue

                    aug_image_name = f"{img_path.stem}_aug{aug_idx}{img_path.suffix}"
                    aug_image_path = self.train_images / aug_image_name
                    aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(aug_image_path), aug_image_bgr)

                    aug_label_path = self.train_labels / f"{img_path.stem}_aug{aug_idx}.txt"
                    self.save_yolo_label(aug_label_path, aug_bboxes, aug_labels)

                    augmented_count += 1
                except Exception as e:
                    print(f"\nWarning: Failed to augment {img_path.name}: {str(e)}")
                    continue

        print(f"\n{'='*60}")
        print(f"✓ Augmentation complete!")
        print(f"✓ Original images: {original_count}")
        print(f"✓ Augmented images created: {augmented_count}")
        print(f"✓ Total training images: {original_count + augmented_count}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    augmenter = BallDataAugmenter()
    augmenter.augment_dataset(multiplier=3, backup=True)
