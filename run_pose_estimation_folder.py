"""
STEP2: Pose Estimation (THIS FILE)
 Input: extracted_frames/ folder with video frames
 Process: Detect human pose landmarks using MediaPipe
 Output: pose_results/ folder with pose data (CSV + JSON)

PURPOSE: Runs pose estimation on all extracted frames using MediaPipe's PoseLandmarker model.
Detects 33 body landmarks per frame and exports results as CSV files for downstream analysis.
"""

import cv2
import csv
import re
from pathlib import Path
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# 0. MODEL SETUP
# =========================

MODEL_PATH = "pose_landmarker_full.task"

def download_model(model_path):
    """Download the pose landmarker model from MediaPipe's official source."""
    import urllib.request
    
    if os.path.exists(model_path):
        print(f"✓ Model already exists: {model_path}")
        return True
    
    print(f"Downloading model to {model_path}...")
    
    # Try multiple URLs
    urls = [
        "https://storage.googleapis.com/mediapipe-models/vision/pose_landmarker/float_wrist/1/pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/vision/pose_landmarker/float_wrist/latest/pose_landmarker_full.task",
    ]
    
    for url in urls:
        try:
            print(f"  Trying: {url}")
            urllib.request.urlretrieve(url, model_path)
            print(f"✓ Model downloaded successfully!")
            return True
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    print(f"\n✗ Could not download model automatically.")
    print(f"Please download manually from:")
    print(f"https://developers.google.com/mediapipe/solutions/vision/pose_landmarker")
    print(f"And save it as: {model_path}")
    return False


if not download_model(MODEL_PATH):
    print("\nModel file required. Exiting...")
    exit(1)


# =========================
# 1. PATH SETTINGS
# =========================

VIDEO_FOLDER = Path("downloaded_videos")
OUTPUT_FOLDER = Path("pose_results")
MODEL_PATH = "pose_landmarker_full.task"

OUTPUT_FOLDER.mkdir(exist_ok=True)


# =========================
# 2. FILENAME PARSING
# =========================
# Example filename:
# gBR_sBM_c01_d04_mBR0_ch01.mp4

filename_pattern = re.compile(
    r"(?P<genre>g[A-Za-z]+)_"
    r"(?P<section>s[A-Za-z]+)_"
    r"(?P<camera>c\d+)_"
    r"(?P<dance_id>d\d+)_"
    r"(?P<movement>m[A-Za-z0-9]+)_"
    r"(?P<channel>ch\d+)\.mp4"
)


def parse_filename(video_name):
    match = filename_pattern.match(video_name)

    if not match:
        return None

    return match.groupdict()


# =========================
# 3. CREATE POSE LANDMARKER FUNCTION
# =========================

def create_landmarker():
    """Create a new pose landmarker instance for video processing."""
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False
    )
    return vision.PoseLandmarker.create_from_options(options)


# =========================
# 4. PROCESS ONE VIDEO
# =========================

def process_video(video_path):
    # Create a new landmarker instance for this video
    landmarker = create_landmarker()
    metadata = parse_filename(video_path.name)

    if metadata is None:
        print("Skipping file with unexpected filename:", video_path.name)
        return

    print("\nProcessing:", video_path.name)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Could not open video:", video_path.name)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        print("Could not read FPS:", video_path.name)
        cap.release()
        return

    output_csv = OUTPUT_FOLDER / (video_path.stem + "_pose.csv")

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        header = [
            "video_name",
            "genre",
            "section",
            "camera",
            "dance_id",
            "movement",
            "channel",
            "frame_index",
            "timestamp_ms",
            "landmark_id",
            "x",
            "y",
            "z",
            "visibility",
            "presence"
        ]

        writer.writerow(header)

        frame_index = 0
        last_timestamp_ms = -1

        while True:
            success, frame = cap.read()

            if not success:
                break

            # Ensure monotonically increasing timestamps for MediaPipe
            timestamp_ms = max(last_timestamp_ms + 1, int((frame_index / fps) * 1000))
            last_timestamp_ms = timestamp_ms

            # OpenCV uses BGR, but MediaPipe needs RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if len(result.pose_landmarks) > 0:
                pose_landmarks = result.pose_landmarks[0]

                for landmark_id, landmark in enumerate(pose_landmarks):
                    writer.writerow([
                        video_path.name,
                        metadata["genre"],
                        metadata["section"],
                        metadata["camera"],
                        metadata["dance_id"],
                        metadata["movement"],
                        metadata["channel"],
                        frame_index,
                        timestamp_ms,
                        landmark_id,
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility,
                        landmark.presence
                    ])

            frame_index += 1

    cap.release()
    landmarker.close()

    print("Saved:", output_csv)


# =========================
# 5. PROCESS ALL VIDEOS
# =========================

video_paths = sorted(VIDEO_FOLDER.glob("*.mp4"))

print("Found", len(video_paths), "videos.")

for video_path in video_paths:
    process_video(video_path)

print("\n✓ Finished pose estimation for all videos.")
print(f"Results saved to: {OUTPUT_FOLDER}/")