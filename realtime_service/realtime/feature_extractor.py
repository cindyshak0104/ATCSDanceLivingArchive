"""
STEP6: Frontend ver2 - Real-time Service
 WebSocket Routes
 Feature Extraction (THIS FILE)
 Real-time prediction and feedback
 Motion buffering and sequence analysis

PURPOSE: Extracts features from live pose stream for real-time emotion prediction.
Buffers recent poses and generates features compatible with STEP3.2 (emotion model).
Provides real-time feedback on emotional content of user's dance.
"""

"""
Feature Extractor: Convert WebSocket landmarks to ML model input features
"""

import numpy as np
from typing import List, Dict, Optional


class LandmarkNormalizer:
    """Normalize landmarks to consistent scale and handle missing data."""

    @staticmethod
    def normalize_landmarks(landmarks: List[Dict]) -> np.ndarray:
        """
        Convert landmark list to normalized feature vector.
        
        Input: List of dicts with keys: name, x, y, visibility
        Output: Numpy array of shape (12, 3) with [x, y, visibility] per landmark
        
        Expected order: shoulders, elbows, wrists, hips, knees, ankles
        """
        if not landmarks or len(landmarks) == 0:
            return None

        # Sort landmarks by expected body order for consistent feature extraction
        landmark_order = [
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]

        # Create lookup dict
        landmark_dict = {lm["name"]: lm for lm in landmarks}

        # Build feature vector
        features = []
        for name in landmark_order:
            if name in landmark_dict:
                lm = landmark_dict[name]
                features.append([lm["x"], lm["y"], lm.get("visibility", 1.0)])
            else:
                # Missing landmark - use zero with low visibility
                features.append([0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)  # Shape: (12, 3)

    @staticmethod
    def extract_motion_vector(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Extract motion between two consecutive frames.
        Returns velocity vector: frame2 - frame1
        """
        if frame1 is None or frame2 is None:
            return None

        # Flatten to compare positions only (ignore visibility)
        pos1 = frame1[:, :2].flatten()  # Shape: (24,) - 12 joints * 2 coords
        pos2 = frame2[:, :2].flatten()

        motion = pos2 - pos1
        return motion


def extract_features_from_buffer(motion_buffer) -> Optional[np.ndarray]:
    """
    Extract ML-ready features from motion buffer.
    Used by Layer 1 (PoseMatcher) and Layer 2 (EmotionPredictor).
    
    Args:
        motion_buffer: MotionBuffer instance with frames
        
    Returns:
        Feature vector suitable for ML models, or None if not enough data
    """
    frames = motion_buffer.get_recent_frames()

    if not frames or len(frames) < 3:
        return None

    # Get latest frame
    latest_frame = frames[-1]

    if latest_frame is None or len(latest_frame) == 0:
        return None

    # Normalize landmarks to feature array
    latest_features = LandmarkNormalizer.normalize_landmarks(latest_frame)

    if latest_features is None:
        return None

    # Return flattened features (66 dimensions: 12 joints * 3 coords + motion)
    return latest_features.flatten()


def extract_motion_features_from_buffer(motion_buffer) -> Optional[np.ndarray]:
    """
    Extract motion features (velocities) from buffer.
    Used for temporal analysis.
    
    Args:
        motion_buffer: MotionBuffer instance
        
    Returns:
        Array of motion vectors or None if insufficient data
    """
    frames = motion_buffer.get_recent_frames()

    if not frames or len(frames) < 2:
        return None

    motion_vectors = []

    for i in range(1, len(frames)):
        frame1 = LandmarkNormalizer.normalize_landmarks(frames[i - 1])
        frame2 = LandmarkNormalizer.normalize_landmarks(frames[i])

        if frame1 is not None and frame2 is not None:
            motion = LandmarkNormalizer.extract_motion_vector(frame1, frame2)
            if motion is not None:
                motion_vectors.append(motion)

    if len(motion_vectors) == 0:
        return None

    # Return mean motion (aggregated velocity pattern)
    return np.mean(motion_vectors, axis=0)


def create_pose_csv_format(landmarks: List[Dict]) -> str:
    """
    Create CSV-style data from landmarks for compatibility with
    Layer 2 (EmotionPredictor) which expects pose CSV files.
    
    This is a fallback format if needed.
    """
    lines = []
    for lm in landmarks:
        name = lm.get("name", "unknown")
        x = lm.get("x", 0)
        y = lm.get("y", 0)
        visibility = lm.get("visibility", 1.0)
        lines.append(f"{name},{x:.4f},{y:.4f},{visibility:.4f}")

    return "\n".join(lines)
