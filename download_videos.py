"""
STEP1: Dataset Preparation
 1.1: Download Videos (THIS FILE)
 1.2: Extract Frames
 1.3: Convert to Stick Figures

PURPOSE: Downloads video files from URLs specified in a CSV file.
This is the first step in the pipeline to acquire raw video data for processing.
"""

import csv
import os
import requests
from urllib.parse import urlparse

csv_file = "refined_2M_sBM_one_ch01_ch10_per_genre.csv"
output_folder = "downloaded_videos"

os.makedirs(output_folder, exist_ok=True)

with open(csv_file, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        url = row["URL"]   # change this if your column name is different

        filename = os.path.basename(urlparse(url).path)

        if filename == "":
            filename = row["VIDEO_NAME"] + ".mp4"   # change if needed

        output_path = os.path.join(output_folder, filename)

        if os.path.exists(output_path):
            print("Already downloaded:", filename)
            continue

        print("Downloading:", filename)

        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open(output_path, "wb") as video_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        video_file.write(chunk)

            print("Saved:", output_path)
        else:
            print("Failed:", url, response.status_code)