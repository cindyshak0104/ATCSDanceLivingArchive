"""
STEP6: Frontend ver2 - Real-time Service
 WebSocket Routes
 Feature Extraction
 Motion Buffering
 Next Movement Prediction (THIS FILE)

PURPOSE: Predicts upcoming dance poses and provides real-time feedback.
Uses STEP3.1 (body similarity) and STEP3.2 (emotion prediction) for live suggestions.
Analyzes pose sequence trends to anticipate next movements.
Provides corrective feedback for dance performance improvement.
"""

"""
Next Movement Predictor: Analyzes motion buffer and predicts the next movement.

Integrates Layer 1 (PoseMatcher) and Layer 2 (EmotionPredictor) models.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from realtime_service.realtime.motion_buffer import MotionBuffer
from realtime_service.realtime.feature_extractor import (
    extract_features_from_buffer,
    extract_motion_features_from_buffer,
)
from realtime_service.realtime.archive_interface import ArchiveInterface

# Add project root to path for model imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class NextMovementPredictor:
    """
    Predicts the next movement based on recent motion history.

    Pipeline:
    1. Extract motion features from buffer
    2. Layer 1: PoseMatcher finds physically similar archive poses
    3. Layer 2: EmotionPredictor scores candidates by emotion
    4. Transition Model: Select likely next movement
    5. Return top-scoring prediction with stick figures
    """

    def __init__(self):
        """Initialize predictor and load models."""
        self.min_frames_required = 5
        self.previous_match = None
        self.archive = ArchiveInterface("stick_images2")
        self.layer1_matcher = None
        self.layer2_predictor = None

        # Attempt to load models (graceful fallback if missing)
        self._load_models()

    def _load_models(self):
        """Load Layer 1 and Layer 2 models if available."""
        try:
            print("[Predictor] Loading Layer 1 PoseMatcher...")
            from models.ml_layer1.match_pose import PoseMatcher

            self.layer1_matcher = PoseMatcher()
            print("[Predictor] Layer 1 PoseMatcher loaded")
        except Exception as e:
            print(f"[Predictor] Warning: Could not load Layer 1: {e}")
            self.layer1_matcher = None

        try:
            print("[Predictor] Loading Layer 2 EmotionPredictor...")
            from models.ml_layer2.predict_emotion import EmotionPredictor

            self.layer2_predictor = EmotionPredictor(
                model_path="saved_models/emotion_model_best.pt",
                metadata_path="saved_models/emotion_model_metadata.json",
                device="cpu",
            )
            print("[Predictor] Layer 2 EmotionPredictor loaded")
        except Exception as e:
            print(f"[Predictor] Warning: Could not load Layer 2: {e}")
            self.layer2_predictor = None

    def predict(self, motion_buffer: MotionBuffer) -> Dict[str, Any]:
        """
        Generate a prediction for the next movement.

        Args:
            motion_buffer: MotionBuffer with recent pose frames

        Returns:
            Prediction dict with movement ID, confidence scores, and stick figures
        """

        # Check if buffer is ready
        if not motion_buffer.is_ready(self.min_frames_required):
            return self._warming_up_response()

        # Get recent frames for analysis
        frames = motion_buffer.get_recent_frames()

        # Layer 1: Find physically similar archive poses
        physical_score = 0.5
        current_match = None
        layer1_matches = None

        if self.layer1_matcher and frames:
            try:
                # Extract features from latest frame
                features = extract_features_from_buffer(motion_buffer)

                if features is not None:
                    # TODO: Layer 1 integration point
                    # This is where PoseMatcher.match() is called
                    # For now, use a mock score
                    physical_score = np.random.uniform(0.65, 0.95)
                    current_match = "clip_001"
                    print("[Layer1] Physical score:", physical_score)
            except Exception as e:
                print(f"[Predictor] Layer 1 error: {e}")

        # Layer 2: Predict emotion if Layer 2 is available
        emotion_score = 0.5
        emotion = "neutral"
        memory_tag = "unknown"

        if self.layer2_predictor and frames:
            try:
                # TODO: Layer 2 integration point
                # This is where EmotionPredictor.predict() is called with motion features
                # For now, use mock scores
                emotion_score = np.random.uniform(0.60, 0.90)
                emotion = np.random.choice(
                    ["powerful", "fluid", "grounded", "expansive", "tender"]
                )
                memory_tag = np.random.choice(
                    ["resistance", "flow", "balance", "reach", "root"]
                )
                print(f"[Layer2] Emotion: {emotion}, Score: {emotion_score}")
            except Exception as e:
                print(f"[Predictor] Layer 2 error: {e}")

        # Transition Model: Predict next movement
        transition_score = 0.75
        predicted_next = "clip_002"

        # Calculate final confidence score
        confidence = (
            physical_score * 0.40
            + emotion_score * 0.25
            + transition_score * 0.35
        )

        # Get stick figure frames for predicted movement
        stick_frames = self.archive.get_movement_frames(predicted_next)
        if stick_frames is None:
            stick_frames = []

        self.previous_match = current_match

        return {
            "type": "prediction",
            "status": "ok",
            "current_matched_movement_id": current_match,
            "predicted_movement_id": predicted_next,
            "stick_figure_path": f"/stick_images2/{predicted_next}/",
            "stick_figure_frames": stick_frames,
            "confidence": round(float(confidence), 3),
            "physical_score": round(float(physical_score), 3),
            "emotion_score": round(float(emotion_score), 3),
            "transition_score": round(float(transition_score), 3),
            "emotion": emotion,
            "memory_tag": memory_tag,
        }

    def _warming_up_response(self) -> Dict[str, Any]:
        """Return warming-up response when buffer isn't ready."""
        return {
            "type": "prediction",
            "status": "warming_up",
            "current_matched_movement_id": None,
            "predicted_movement_id": None,
            "stick_figure_path": None,
            "stick_figure_frames": [],
            "confidence": None,
            "physical_score": None,
            "emotion_score": None,
            "transition_score": None,
            "emotion": "calibrating",
            "memory_tag": "calibrating",
        }
