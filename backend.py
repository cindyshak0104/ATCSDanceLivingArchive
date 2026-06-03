"""
STEP4: Backend API (THIS FILE)
 Exposes REST endpoints for pose matching and emotion prediction
 Integrates STEP3 ML models (body similarity + emotional similarity)
 Serves STEP5 & STEP6 Frontend applications
 Provides WebSocket routes for real-time predictions

PURPOSE: FastAPI backend server that:
- Loads pose matching model (STEP3.1 - Body Similarity)
- Handles image uploads for pose extraction
- Performs pose matching against archive database
- Integrates emotion prediction (STEP3.2 - Emotional Similarity)
- Returns ranked dance pose matches with visualizations
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from mediapipe.tasks.python.vision import pose_landmarker
from mediapipe.tasks.python.vision.core import image as mp_image

from models.ml_layer1.match_pose import PoseMatcher as DancePoseMatcher


# Import real-time WebSocket router
from realtime_service.realtime.websocket_routes import router as realtime_router


APP = FastAPI()

APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include real-time WebSocket routes
APP.include_router(realtime_router)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "pose_landmarker_full.task"
TOP_K_MATCHES = 12
DEFAULT_FPS = 24

# This allows React to load local images through URLs such as:
# http://localhost:8000/static/stick_images2/.../frame_0000.png
#
# For a prototype, mounting ROOT is convenient.
# Later, you can restrict this to only the stick_images2 folder.
APP.mount("/static", StaticFiles(directory=ROOT), name="static")

if not MODEL_PATH.exists():
    raise RuntimeError(f"MediaPipe model not found: {MODEL_PATH}")

POSE_LANDMARKER = pose_landmarker.PoseLandmarker.create_from_model_path(str(MODEL_PATH))


def build_pose_database_df_from_folder(folder_path: Path, stick_root: Path) -> pd.DataFrame:
    csv_paths = sorted(folder_path.glob("*_pose.csv"))

    if not csv_paths:
        raise ValueError(
            f"No full landmark CSV files found in {folder_path}. "
            "Expect files named '*_pose.csv' in pose_results or pose_results_opencv."
        )

    records = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        required_columns = {"frame_index", "landmark_id", "x", "y", "z", "visibility"}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            continue

        base_name = csv_path.stem

        if base_name.endswith("_pose"):
            base_name = base_name[: -len("_pose")]

        grouped = df.groupby("frame_index")

        for frame_index, frame_df in grouped:
            if frame_df["landmark_id"].nunique() != 33:
                continue

            arr = np.full((33, 4), np.nan, dtype=np.float32)

            for _, row in frame_df.iterrows():
                landmark_id = int(row["landmark_id"])

                if 0 <= landmark_id < 33:
                    arr[landmark_id] = [
                        float(row["x"]),
                        float(row["y"]),
                        float(row["z"]),
                        float(row["visibility"]),
                    ]

            if np.isnan(arr).any():
                continue

            stick_figure_path = stick_root / base_name / f"frame_{int(frame_index):04d}.png"

            row = {
                "pose_id": f"{base_name}_frame_{int(frame_index):04d}",
                "movement_id": base_name,
                "frame_index": int(frame_index),
                "stick_figure_path": str(stick_figure_path.resolve()),
            }

            for i in range(33):
                row[f"lm{i}_x"] = float(arr[i, 0])
                row[f"lm{i}_y"] = float(arr[i, 1])
                row[f"lm{i}_z"] = float(arr[i, 2])
                row[f"lm{i}_visibility"] = float(arr[i, 3])

            records.append(row)

    if not records:
        raise ValueError(f"No valid 33-landmark frames were found in {folder_path}.")

    return pd.DataFrame(records)


def load_matcher(
    archive_path: str,
    stick_images_root: str,
    metric: str,
    min_visibility: float,
) -> DancePoseMatcher:
    matcher = DancePoseMatcher(metric=metric, min_visibility=min_visibility)

    archive_obj = Path(archive_path)

    if archive_obj.is_dir():
        df = build_pose_database_df_from_folder(archive_obj, Path(stick_images_root))
        matcher.fit_from_dataframe(df)
    else:
        matcher.fit_from_csv(archive_obj)

    return matcher


def file_to_landmarks(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Unable to decode image")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp_image.Image(mp_image.ImageFormat.SRGB, rgb)

    results = POSE_LANDMARKER.detect(mp_img)

    if not results.pose_landmarks:
        raise ValueError("No pose detected in the image")

    return results.pose_landmarks[0]


def image_path_to_data_uri(path: str) -> Optional[str]:
    if not path:
        return None

    image_path = Path(path)

    if not image_path.exists():
        return None

    image_bytes = image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")

    return f"data:image/png;base64,{encoded}"


def extract_movement_id_from_pose_id(pose_id: str) -> str:
    """
    Converts:
        gBR_sBM_c01_d04_mBR0_ch01_frame_0032

    Into:
        gBR_sBM_c01_d04_mBR0_ch01
    """

    if "_frame_" in pose_id:
        return pose_id.split("_frame_")[0]

    return pose_id


def path_to_static_url(path: Path, request: Request) -> Optional[str]:
    """
    Converts a local file path into a URL React can display.

    Example:
        /Users/.../project/stick_images2/clip/frame_0000.png

    Becomes:
        http://localhost:8000/static/stick_images2/clip/frame_0000.png
    """

    try:
        resolved_path = path.resolve()
        relative_path = resolved_path.relative_to(ROOT)
    except ValueError:
        return None

    base_url = str(request.base_url).rstrip("/")
    url_path = str(relative_path).replace("\\", "/")

    return f"{base_url}/static/{url_path}"


def get_stick_frames_for_movement(
    movement_id: str,
    stick_root: Path,
    request: Request,
) -> list[str]:
    """
    Returns all frame URLs for one movement folder.

    Expected folder structure:

    stick_images2/
    └── gBR_sBM_c01_d04_mBR0_ch01/
        ├── frame_0000.png
        ├── frame_0001.png
        ├── frame_0002.png
        └── ...
    """

    movement_folder = stick_root / movement_id

    if not movement_folder.exists() or not movement_folder.is_dir():
        return []

    frame_paths = sorted(
        list(movement_folder.glob("*.png"))
        + list(movement_folder.glob("*.jpg"))
        + list(movement_folder.glob("*.jpeg"))
    )

    frame_urls = []

    for frame_path in frame_paths:
        frame_url = path_to_static_url(frame_path, request)

        if frame_url:
            frame_urls.append(frame_url)

    return frame_urls


@APP.get("/")
def health_check():
    return {
        "status": "Backend is running",
        "match_route": "/match-pose",
        "static_route": "/static",
    }


@APP.post("/match-pose")
async def match_pose(
    request: Request,
    image: UploadFile = File(...),
    archive_source: str = Form("Full landmark folder"),
    archive_path: str = Form("pose_results"),
    stick_images_root: str = Form("stick_images2"),
    metric: str = Form("cosine"),
    min_visibility: float = Form(0.3),
    show_skeleton: bool = Form(True),
    show_stick_figure_panel: bool = Form(True),
):
    try:
        image_bytes = await image.read()
        pose_landmarks = file_to_landmarks(image_bytes)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Pose detection failed: {str(exc)}",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected image-processing error: {str(exc)}",
        )

    archive_full_path = Path(archive_path)

    if not archive_full_path.is_absolute():
        archive_full_path = ROOT / archive_full_path

    stick_full_path = Path(stick_images_root)

    if not stick_full_path.is_absolute():
        stick_full_path = ROOT / stick_full_path

    if not archive_full_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Archive path not found: {archive_full_path}",
        )

    if not stick_full_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Stick image root not found: {stick_full_path}",
        )

    try:
        matcher = load_matcher(
            str(archive_full_path),
            str(stick_full_path),
            metric,
            min_visibility,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Matcher loading failed: {str(exc)}",
        )

    try:
        matches = matcher.match_top_k(pose_landmarks, k=TOP_K_MATCHES)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pose matching failed: {str(exc)}",
        )

    results = []

    for match in matches:
        pose_id = match.pose_id
        movement_id = extract_movement_id_from_pose_id(pose_id)

        matched_frame_path = Path(match.stick_figure_path)
        matched_frame_data_uri = image_path_to_data_uri(match.stick_figure_path)
        matched_frame_url = path_to_static_url(matched_frame_path, request)

        stick_frames = get_stick_frames_for_movement(
            movement_id=movement_id,
            stick_root=stick_full_path,
            request=request,
        )

        if hasattr(match, "similarity"):
            similarity_score = float(match.similarity)
        else:
            similarity_score = float(1 / (1 + match.distance))

        result = {
            # General identity fields
            "id": pose_id,
            "pose_id": pose_id,
            "movement_id": movement_id,
            "title": movement_id,

            # Matching scores
            "distance": float(match.distance),
            "similarity_score": similarity_score,
            "final_score": similarity_score,
            "score": similarity_score,

            # Single-frame preview fields
            "stick_figure_path": match.stick_figure_path,
            "image": matched_frame_data_uri,
            "image_url": matched_frame_data_uri,
            "previewFrame": matched_frame_url,

            # Sequence-flow fields
            "fps": DEFAULT_FPS,
            "stickFrames": stick_frames,

            # Optional metadata placeholders
            "emotion": "unlabeled",
            "memory_tag": "unlabeled",
        }

        results.append(result)

    return {
        "matches": results,
        "num_matches": len(results),
    }