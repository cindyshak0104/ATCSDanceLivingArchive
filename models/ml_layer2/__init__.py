"""
STEP3: ML Models - Layer 2 (Emotional Similarity)
Package initialization for emotion model modules.
"""

"""
Layer 2: Emotion and Memory Similarity Prediction

A multi-task neural network that predicts emotion vectors and memory tags
from pose-derived features. Used to re-rank Layer 1 (body similarity) matches
by emotional similarity.
"""

from .emotion_model import EmotionNet, EmotionNetSimple
from .extract_emotion_features import process_all_csvs
from .predict_emotion import EmotionPredictor, batch_predict
from .rerank_matches import rerank_by_emotion, rerank_with_confidence

__all__ = [
    'EmotionNet',
    'EmotionNetSimple',
    'process_all_csvs',
    'EmotionPredictor',
    'batch_predict',
    'rerank_by_emotion',
    'rerank_with_confidence',
]
