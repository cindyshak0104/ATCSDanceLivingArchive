"""
STEP3: ML Models - Layer 2 (Emotional Similarity)
UTILITY: Rerank Matches by Emotion (THIS FILE)

PURPOSE: Reranks pose matches based on emotional similarity.
Takes initial pose matches (from Layer 1 body similarity) and reranks them using emotion predictions.
Combines body and emotional similarity for better overall matching.
Used in STEP4 backend for final result ranking.
"""

"""
Layer 1 + Layer 2 integration: Rerank physically similar matches by emotional similarity.

The reranking formula combines:
- Layer 1 body similarity (from pose-based matching)
- Layer 2 emotion similarity (from emotion vector comparison)

Final score = alpha * body_similarity + (1 - alpha) * emotion_similarity
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import json


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
    
    Returns:
        Cosine similarity score in [-1, 1], typically normalized to [0, 1]
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    similarity = dot_product / (norm_a * norm_b)
    return (similarity + 1) / 2


def euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean distance-based similarity.
    Closer vectors have higher similarity.
    
    Args:
        a: First vector
        b: Second vector
    
    Returns:
        Similarity score in [0, 1]
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    
    distance = np.linalg.norm(a - b)
    similarity = np.exp(-distance)
    return similarity


def rerank_by_emotion(
    layer1_matches: List[Dict],
    query_emotion_vector: np.ndarray,
    archive_emotion_vectors: Dict[str, np.ndarray],
    alpha: float = 0.6,
    similarity_metric: str = 'cosine'
) -> List[Dict]:
    """
    Rerank Layer 1 matches by adding emotional similarity.
    
    Final score = alpha * body_similarity + (1 - alpha) * emotion_similarity
    
    Args:
        layer1_matches: List of dicts from Layer 1
        query_emotion_vector: Emotion vector for the query movement
        archive_emotion_vectors: Dictionary mapping movement IDs to emotion vectors
        alpha: Weight for body similarity
        similarity_metric: 'cosine' or 'euclidean'
    
    Returns:
        Reranked list of matches with emotion scores
    """
    
    if similarity_metric == 'cosine':
        similarity_fn = cosine_similarity
    elif similarity_metric == 'euclidean':
        similarity_fn = euclidean_similarity
    else:
        raise ValueError(f"Unknown similarity metric: {similarity_metric}")
    
    reranked = []
    
    for match in layer1_matches:
        movement_id = match["movement_id"]
        body_score = match["body_similarity"]
        
        if movement_id not in archive_emotion_vectors:
            emotion_score = 0.5
        else:
            candidate_emotion = archive_emotion_vectors[movement_id]
            emotion_score = similarity_fn(query_emotion_vector, candidate_emotion)
        
        final_score = alpha * body_score + (1 - alpha) * emotion_score
        
        reranked.append({
            "movement_id": movement_id,
            "body_similarity": body_score,
            "emotion_similarity": emotion_score,
            "final_score": final_score
        })
    
    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    
    return reranked


def rerank_with_confidence(
    layer1_matches: List[Dict],
    query_emotion_vector: np.ndarray,
    query_emotion_confidence: float,
    archive_emotion_vectors: Dict[str, np.ndarray],
    archive_emotion_confidence: Dict[str, float],
    alpha: float = 0.6,
    similarity_metric: str = 'cosine'
) -> List[Dict]:
    """
    Advanced reranking that considers confidence scores.
    
    When emotion prediction confidence is low, rely more on body similarity.
    When it's high, rely more on emotion similarity.
    
    Args:
        layer1_matches: Layer 1 matches
        query_emotion_vector: Query emotion vector
        query_emotion_confidence: Confidence in query emotion prediction
        archive_emotion_vectors: Archive emotion vectors
        archive_emotion_confidence: Confidence in archive emotion predictions
        alpha: Base weight for body similarity
        similarity_metric: 'cosine' or 'euclidean'
    
    Returns:
        Reranked matches with confidence-adjusted scores
    """
    
    if similarity_metric == 'cosine':
        similarity_fn = cosine_similarity
    elif similarity_metric == 'euclidean':
        similarity_fn = euclidean_similarity
    else:
        raise ValueError(f"Unknown similarity metric: {similarity_metric}")
    
    reranked = []
    
    for match in layer1_matches:
        movement_id = match["movement_id"]
        body_score = match["body_similarity"]
        
        candidate_confidence = archive_emotion_confidence.get(movement_id, 0.5)
        
        if movement_id not in archive_emotion_vectors:
            emotion_score = 0.5
        else:
            candidate_emotion = archive_emotion_vectors[movement_id]
            emotion_score = similarity_fn(query_emotion_vector, candidate_emotion)
        
        avg_confidence = (query_emotion_confidence + candidate_confidence) / 2
        adjusted_alpha = alpha + (1 - alpha) * (1 - avg_confidence)
        
        final_score = adjusted_alpha * body_score + (1 - adjusted_alpha) * emotion_score
        
        reranked.append({
            "movement_id": movement_id,
            "body_similarity": body_score,
            "emotion_similarity": emotion_score,
            "query_emotion_confidence": query_emotion_confidence,
            "candidate_emotion_confidence": candidate_confidence,
            "adjusted_alpha": adjusted_alpha,
            "final_score": final_score
        })
    
    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    
    return reranked


def save_reranked_results(
    reranked_matches: List[Dict],
    output_path: str
):
    """Save reranked results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(reranked_matches, f, indent=2)
    print(f"Saved reranked results to {output_path}")


if __name__ == "__main__":
    # Example usage
    layer1_matches = [
        {"movement_id": "move_045", "body_similarity": 0.92},
        {"movement_id": "move_102", "body_similarity": 0.88},
        {"movement_id": "move_087", "body_similarity": 0.85},
        {"movement_id": "move_211", "body_similarity": 0.82},
        {"movement_id": "move_034", "body_similarity": 0.79},
    ]
    
    query_emotion = np.array([0.4, 0.7, 0.8, 0.2, 0.3])
    
    archive_emotions = {
        "move_045": np.array([0.5, 0.8, 0.7, 0.3, 0.2]),
        "move_102": np.array([0.8, 0.9, 0.2, 0.1, 0.9]),
        "move_087": np.array([0.6, 0.5, 0.8, 0.7, 0.3]),
        "move_211": np.array([0.3, 0.6, 0.9, 0.1, 0.4]),
        "move_034": np.array([0.4, 0.7, 0.75, 0.25, 0.35]),
    }
    
    reranked = rerank_by_emotion(
        layer1_matches,
        query_emotion,
        archive_emotions,
        alpha=0.6,
        similarity_metric='cosine'
    )
    
    print("Reranked matches (alpha=0.6, cosine similarity):")
    print("=" * 80)
    for i, match in enumerate(reranked, 1):
        print(f"{i}. {match['movement_id']}")
        print(f"   Body similarity: {match['body_similarity']:.3f}")
        print(f"   Emotion similarity: {match['emotion_similarity']:.3f}")
        print(f"   Final score: {match['final_score']:.3f}")
        print()
