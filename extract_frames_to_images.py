"""
STEP1: Dataset Preparation
 1.1: Download Videos
 1.2: Extract Frames (THIS FILE)
 1.3: Convert to Stick Figures

PURPOSE: Extracts individual frames from downloaded video files and saves them as images.
Input: downloaded_videos/ folder
Output: extracted_frames/ folder with frame images
"""

#!/usr/bin/env python3
"""
Extract frames from videos in downloaded_videos folder and save them as images.
This prepares the data for pose estimation.
"""

import cv2
from pathlib import Path
import os
from tqdm import tqdm

VIDEOS_FOLDER = Path("downloaded_videos")
FRAMES_FOLDER = Path("extracted_frames")

# Create frames output folder
FRAMES_FOLDER.mkdir(exist_ok=True)

video_files = sorted(VIDEOS_FOLDER.glob("*.mp4"))

print(f"Found {len(video_files)} videos to process\n")

for video_path in video_files:
    print(f"Processing: {video_path.name}")
    
    # Create a subfolder for frames from this video
    video_frames_folder = FRAMES_FOLDER / video_path.stem
    video_frames_folder.mkdir(exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"  ✗ Could not open video")
        continue
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_index = 0
    saved_count = 0
    
    while True:
        success, frame = cap.read()
        
        if not success:
            break
        
        # Save every frame (or adjust sampling if needed)
        frame_path = video_frames_folder / f"frame_{frame_index:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        saved_count += 1
        
        frame_index += 1
    
    cap.release()
    
    print(f"  ✓ Extracted {saved_count} frames to {video_frames_folder}")
    print(f"    Total frames: {total_frames}, FPS: {fps}\n")

print(f"✓ Frame extraction complete!")
print(f"Frames saved to: {FRAMES_FOLDER}/")
print(f"\nTo process with pose estimation, ensure pose_landmarker_full.task is available")
