import cv2
import os
from ball_detection import BallDetector

# Local dataset path
dataset_path = "Cricket Ball Detection.v1i.yolov8"
print("Using local dataset at:", dataset_path)

# Define paths for training images
train_images_dir = os.path.join(dataset_path, "train", "images")
valid_images_dir = os.path.join(dataset_path, "valid", "images")

# Load training images
def load_images_from_folder(folder, limit=50):
    imgs = []
    if not os.path.exists(folder):
        print(f"Warning: Folder {folder} does not exist")
        return imgs
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(folder) if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    for i, filename in enumerate(image_files):
        if i >= limit: 
            break
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            imgs.append(img)
        else:
            print(f"Warning: Could not load image {img_path}")
    
    print(f"Loaded {len(imgs)} images from {folder}")
    return imgs

# Load training images (positive samples - images with balls)
positive_images = load_images_from_folder(train_images_dir, limit=100)

# For negative samples, we'll use some images from validation set
# In a real scenario, you might want to create a separate negative dataset
negative_images = load_images_from_folder(valid_images_dir, limit=50)

# Train + init detectors
detector = BallDetector()
if positive_images and negative_images:
    detector.train(positive_images, negative_images)
    print("SVM trained with {} positives and {} negatives".format(len(positive_images), len(negative_images)))
else:
    print("Skipping SVM training (no dataset found)")

# Load and train YOLO model on the local dataset
detector.load_yolo(dataset_path)

# Test detection on a sample image
if positive_images:
    test_img = positive_images[0]
    detections = detector.detect(test_img)

    for (x, y, w, h) in detections:
        cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Ball Detection", test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No test images found.")
