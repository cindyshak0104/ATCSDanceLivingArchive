"""
STEP3: ML Models Package
Exports main model interfaces for use in STEP4 (backend.py) and STEP6 (realtime_service)
"""

"""
Dance Living Archive ML Layers

Multi-layer architecture for suggesting next movements:

Layer 1 (Body Similarity) - ml_layer1
    Finds closest pose based on the neural-network pose encoder

Layer 2 (Emotion & Memory) - emotion_encoder.py
    Captures emotional quality and movement memory

Layer 3 (Movement Transitions) - transition_model.py
    Models valid dance transitions between poses

Layer 4 (Choreographic Sequence) - sequence_model.py
    Learns human choreographic ordering patterns

Choreographic Ranker - choreographic_ranker.py
    Combines all 4 layers to rank and suggest best next movement
"""

try:
    from models.dance_pose_matcher import DancePoseMatcher, PoseMatch
except ModuleNotFoundError:
    try:
        from models.ml_layer1.match_pose import PoseMatcher as DancePoseMatcher
    except ModuleNotFoundError:
        DancePoseMatcher = None
    PoseMatch = None

try:
    from models.emotion_encoder import EmotionEncoder
except ModuleNotFoundError:
    EmotionEncoder = None

try:
    from models.transition_model import TransitionModel
except ModuleNotFoundError:
    TransitionModel = None

try:
    from models.sequence_model import SequenceModel
except ModuleNotFoundError:
    SequenceModel = None

try:
    from models.choreographic_ranker import ChoreographicRanker
except ModuleNotFoundError:
    ChoreographicRanker = None

__all__ = [
    "DancePoseMatcher",
    "PoseMatch",
    "EmotionEncoder",
    "TransitionModel",
    "SequenceModel",
    "ChoreographicRanker",
]
