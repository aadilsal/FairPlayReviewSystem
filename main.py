import cv2
import os
from ball_detection import BallDetector
from frame_extractor import extract_frames
from person_detector import detect_persons
from pose_estimator import estimate_pose

videos_folder = "videos"
output_frames_folder = "frame"

# Get all video files in the videos folder
video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
video_files = [f for f in os.listdir(videos_folder) if any(f.lower().endswith(ext) for ext in video_extensions)]

all_frames = {}

print("[INFO] Extracting frames from all videos...")
for video_file in video_files:
    video_path = os.path.join(videos_folder, video_file)
    video_name = os.path.splitext(video_file)[0]
    video_output_folder = os.path.join(output_frames_folder, video_name)
    
    # Check if frames already exist for this video
    if os.path.exists(video_output_folder):
        existing_frames = [f for f in os.listdir(video_output_folder) if f.lower().endswith('.jpg')]
        if existing_frames:
            print(f"Frames already exist for {video_file}, skipping extraction.")
            all_frames[video_name] = [os.path.join(video_output_folder, f) for f in sorted(existing_frames)]
            continue

    os.makedirs(video_output_folder, exist_ok=True)
    frame_paths = extract_frames(video_path, output_folder=video_output_folder)
    all_frames[video_name] = frame_paths
    print(f"Extracted {len(frame_paths)} frames from {video_file} to {video_output_folder}")


print("[INFO] Running person detection and pose estimation on a few frames from vid2...")

vid2_frames = all_frames.get("vid2", [])
for frame_path in vid2_frames:  # just test first 3 frames of vid2
    pose_marker = frame_path + ".pose"
    if os.path.exists(pose_marker):
        print(f"[INFO] Pose already estimated for {frame_path}, skipping.")
        continue

    person_marker = frame_path + ".person"
    if not os.path.exists(person_marker):
        frame = cv2.imread(frame_path)
        frame_with_persons, detections = detect_persons(frame)
        cv2.imwrite(frame_path, frame_with_persons)
        # Create a marker file to indicate person detection is done
        with open(person_marker, "w") as f:
            f.write("person detected")
    else:
        frame_with_persons = cv2.imread(frame_path)

    frame_with_pose, keypoints = estimate_pose(frame_with_persons)
    print(f"[DEBUG] {frame_path} -> {len(keypoints)} persons detected with skeletons")
    cv2.imshow("Pose Estimation", frame_with_pose)
    cv2.imwrite(frame_path, frame_with_pose)
    # Create a marker file to indicate pose estimation is done
    with open(pose_marker, "w") as f:
        f.write("pose estimated")
    cv2.waitKey(0)  # press any key for next frame

cv2.destroyAllWindows()

print("[INFO]  person detection Completed...")



# Load frames for further processing (example: first video)
if all_frames:
    first_video = next(iter(all_frames))
    frames = [cv2.imread(f) for f in all_frames[first_video]]
else:
    frames = []

# video_path = "videos/vid2.mp4"
# print("[INFO] Extracting frames...")
# frame_paths = extract_frames(video_path)
# frames = [cv2.imread(f) for f in frame_paths]

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
