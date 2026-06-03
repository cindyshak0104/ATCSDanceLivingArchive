"""
STEP1: Dataset Preparation
 1.1: Download Videos
 1.2: Extract Frames
 1.3: Convert to Stick Figures (THIS FILE)

PURPOSE: Converts pose estimation results (CSV files) into stick figure visualizations.
Takes pose landmarks and renders them as simple stick figure images for visualization.
Input: pose_results/ folder with CSV files
Output: stick_images2/ folder with stick figure images
"""

import cv2
import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# 1. PATH SETTINGS
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_FOLDER = SCRIPT_DIR / "downloaded_videos"
POSSIBLE_CSV_FOLDERS = [SCRIPT_DIR / "pose_results", SCRIPT_DIR / "pose_results_opencv"]
OUTPUT_FOLDER = SCRIPT_DIR / "stick_images2"

OUTPUT_FOLDER.mkdir(exist_ok=True)

CSV_FOLDER = None
for folder in POSSIBLE_CSV_FOLDERS:
    if folder.exists() and any(folder.glob("*_pose.csv")):
        CSV_FOLDER = folder
        break

if CSV_FOLDER is None:
    CSV_FOLDER = POSSIBLE_CSV_FOLDERS[0]
    print("Warning: no full landmark CSV files found in pose_results or pose_results_opencv.")
    print("If you only have simple pose CSVs in pose_results_simple, run a full pose estimation script first.")

print(f"Looking for pose CSV files in: {CSV_FOLDER}")
print(f"Looking for original videos in: {VIDEO_FOLDER}")


# =========================
# 2. POSE CONNECTIONS
# =========================
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26),
    (25, 27), (26, 28),
    (27, 29), (28, 30),
    (29, 31), (30, 32),
    (27, 31), (28, 32),
]


# =========================
# 3. DRAW ONE FRAME
# =========================
def draw_stick_figure(frame_df, width, height, visibility_threshold=0.5):
    """
    Draw one stick figure on a transparent canvas.
    frame_df = rows for one frame only
    """
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    points = {}

    for _, row in frame_df.iterrows():
        landmark_id = int(row["landmark_id"])
        x = row["x"]
        y = row["y"]

        if pd.isna(x) or pd.isna(y):
            continue

        # Optional visibility filtering
        if "visibility" in frame_df.columns and not pd.isna(row["visibility"]):
            if row["visibility"] < visibility_threshold:
                continue

        # Convert normalized coordinates to pixel coordinates
        px = int(x * width)
        py = int(y * height)

        if 0 <= px < width and 0 <= py < height:
            points[landmark_id] = (px, py)

    # Draw the head as a circle instead of a cluster of point lines.
    head_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    if 0 in points:
        nose = points[0]
        head_radius = 0
        for landmark_id in head_landmarks:
            if landmark_id in points:
                dx = points[landmark_id][0] - nose[0]
                dy = points[landmark_id][1] - nose[1]
                head_radius = max(head_radius, int((dx * dx + dy * dy) ** 0.5))

        if head_radius < 10:
            head_radius = 10
        cv2.circle(canvas, nose, head_radius, (255, 255, 255, 255), 4, cv2.LINE_AA)

    # Draw body and limbs only, skipping the head landmark connections.
    for start, end in POSE_CONNECTIONS:
        if start in head_landmarks and end in head_landmarks:
            continue
        if start in points and end in points:
            cv2.line(canvas, points[start], points[end], (255, 255, 255, 255), 4, cv2.LINE_AA)

    return canvas


# =========================
# 4. PROCESS ONE CSV
# =========================
def process_pose_csv(csv_path):
    """
    Save one PNG per frame from one pose CSV.
    """
    print(f"\nProcessing CSV: {csv_path.name}")

    base_name = csv_path.stem
    if base_name.endswith("_simple_pose"):
        base_name = base_name[:-len("_simple_pose")]
    elif base_name.endswith("_pose"):
        base_name = base_name[:-len("_pose")]
    video_path = VIDEO_FOLDER / f"{base_name}.mp4"

    if not video_path.exists():
        print(f"Original video not found: {video_path.name}")
        return

    # Read original video dimensions and frame count
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path.name}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Read pose CSV
    df = pd.read_csv(csv_path)

    required_columns = {"frame_index", "landmark_id", "x", "y"}
    if not required_columns.issubset(df.columns):
        print(f"Skipping {csv_path.name}: missing required landmark columns {required_columns - set(df.columns)}")
        print("This script requires full landmark CSVs generated by pose estimation into pose_results/")
        return

    if df.empty:
        print(f"CSV is empty: {csv_path.name}")
        return

    grouped = df.groupby("frame_index")

    # Create a subfolder for this video's frames
    video_output_folder = OUTPUT_FOLDER / base_name
    video_output_folder.mkdir(exist_ok=True)

    # Save one image per frame
    for frame_index in range(frame_count):
        if frame_index in grouped.groups:
            frame_df = grouped.get_group(frame_index)
            canvas = draw_stick_figure(frame_df, width, height)
        else:
            # blank transparent image if no pose detected
            canvas = np.zeros((height, width, 4), dtype=np.uint8)

        output_image_path = video_output_folder / f"frame_{frame_index:04d}.png"
        cv2.imwrite(str(output_image_path), canvas)

    print(f"Saved images to: {video_output_folder}")


# =========================
# 5. PROCESS ALL CSV FILES
# =========================
csv_paths = sorted(CSV_FOLDER.glob("*_pose.csv"))

print(f"Found {len(csv_paths)} pose CSV files.")

for csv_path in csv_paths:
    process_pose_csv(csv_path)

print("\nFinished generating stick-figure image sequences.")