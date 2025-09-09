import cv2
import os

def extract_frames(video_path, output_folder="outputs/frames", target_fps=30):
    """
    Extract frames from a video file at a specified frame rate
    
    Args:
        video_path (str): Path to the input video file
        output_folder (str): Directory to save extracted frames
        target_fps (int): Target frames per second for extraction
        
    Returns:
        list: List of paths to extracted frame files
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps
    
    print(f"Video properties:")
    print(f"  Original FPS: {original_fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Target FPS: {target_fps}")
    
    # Calculate frame interval
    frame_interval = max(1, int(round(original_fps / target_fps)))
    expected_frames = total_frames // frame_interval
    
    print(f"  Frame interval: {frame_interval}")
    print(f"  Expected extracted frames: {expected_frames}")
    
    frame_count = 0
    saved_count = 0
    frames = []
    
    print("Extracting frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save only every `frame_interval` frames
        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{saved_count:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            
            # Save frame
            success = cv2.imwrite(frame_path, frame)
            if success:
                frames.append(frame_path)
                saved_count += 1
            else:
                print(f"Warning: Failed to save frame {frame_count}")
        
        frame_count += 1
        
        # Progress update
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    cap.release()
    
    print(f"Frame extraction complete!")
    print(f"  Extracted {saved_count} frames")
    print(f"  Frames saved to: {output_folder}")
    
    return frames


def get_video_info(video_path):
    """
    Get detailed information about a video file
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        dict: Dictionary containing video properties
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
        'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
    }
    
    cap.release()
    return info


def extract_frames_time_range(video_path, start_time, end_time, output_folder="outputs/frames", target_fps=30):
    """
    Extract frames from a specific time range in a video
    
    Args:
        video_path (str): Path to the input video file
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        output_folder (str): Directory to save extracted frames
        target_fps (int): Target frames per second for extraction
        
    Returns:
        list: List of paths to extracted frame files
    """
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps
    
    # Validate time range
    start_time = max(0, start_time)
    end_time = min(duration, end_time)
    
    if start_time >= end_time:
        print("Error: Invalid time range")
        cap.release()
        return []
    
    print(f"Extracting frames from {start_time:.2f}s to {end_time:.2f}s")
    
    # Calculate frame numbers
    start_frame = int(start_time * original_fps)
    end_frame = int(end_time * original_fps)
    frame_interval = max(1, int(round(original_fps / target_fps)))
    
    # Set video position to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = start_frame
    saved_count = 0
    frames = []
    
    while frame_count <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame if it matches the interval
        if (frame_count - start_frame) % frame_interval == 0:
            frame_filename = f"frame_{saved_count:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            
            success = cv2.imwrite(frame_path, frame)
            if success:
                frames.append(frame_path)
                saved_count += 1
        
        frame_count += 1
    
    cap.release()
    
    print(f"Extracted {saved_count} frames from time range {start_time:.2f}s - {end_time:.2f}s")
    return frames


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python frame_extractor.py <video_path> [target_fps] [output_folder]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    target_fps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    output_folder = sys.argv[3] if len(sys.argv) > 3 else "outputs/frames"
    
    # Get video info
    info = get_video_info(video_path)
    if info:
        print("Video Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        print()
    
    # Extract frames
    frames = extract_frames(video_path, output_folder, target_fps)
    print(f"Extraction complete. {len(frames)} frames saved.")