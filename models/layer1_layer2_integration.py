"""
STEP3: ML Models - Integration
PURPOSE: Integrates Layer 1 (Body Similarity) and Layer 2 (Emotional Similarity)

This module coordinates between:
- STEP3.1: Body similarity matching via pose embeddings
- STEP3.2: Emotional similarity filtering and reranking
- STEP4: Backend API that uses both layers

Handles cascade matching: first by body pose, then refines by emotion.
"""

"""
Layer 1 + Layer 2 Integration: Combined Pose and Emotion Matching
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import sys

sys.path.insert(0, 'models/ml_layer2')

from models.ml_layer2.predict_emotion import EmotionPredictor
from models.ml_layer2.rerank_matches import rerank_by_emotion


class Layer1Layer2Matcher:
    """Complete matcher combining Layer 1 (body) and Layer 2 (emotion)."""
    
    def __init__(
        self,
        layer2_model_path: str = "saved_models/emotion_model_best.pt",
        layer2_metadata_path: str = "saved_models/emotion_model_metadata.json",
        device: str = 'cpu',
        alpha: float = 0.6
    ):
        """Initialize the matcher."""
        self.alpha = alpha
        self.device = device
        
        print("Initializing Layer 2 emotion predictor...")
        self.emotion_predictor = EmotionPredictor(
            layer2_model_path,
            layer2_metadata_path,
            device=device
        )
        
        self.emotion_cache = {}
    
    def get_emotion_vector(self, csv_path: str) -> np.ndarray:
        """Get aggregated emotion vector for a CSV file."""
        
        if csv_path in self.emotion_cache:
            return self.emotion_cache[csv_path]
        
        predictions = self.emotion_predictor.predict_from_csv(csv_path)
        aggregated = np.mean(predictions['emotion_vector'], axis=0)
        self.emotion_cache[csv_path] = aggregated
        
        return aggregated
    
    def rerank_layer1_matches(
        self,
        layer1_matches: List[Dict],
        query_emotion_vector: np.ndarray,
        archive_emotion_vectors: Dict[str, np.ndarray],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """Rerank Layer 1 matches by emotion similarity."""
        
        reranked = rerank_by_emotion(
            layer1_matches,
            query_emotion_vector,
            archive_emotion_vectors,
            alpha=self.alpha,
            similarity_metric='cosine'
        )
        
        if top_k is not None:
            reranked = reranked[:top_k]
        
        return reranked
    
    def full_matching_pipeline(
        self,
        query_csv: str,
        layer1_matches: List[Dict],
        archive_csv: str,
        top_k: int = 5,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Complete matching pipeline: Layer 1 + Layer 2.
        """
        
        if verbose:
            print("\n" + "="*80)
            print("LAYER 1 + LAYER 2 INTEGRATION")
            print("="*80)
        
        # Get query emotion vector
        if verbose:
            print(f"\n[1/4] Getting query emotion vector from {query_csv}")
        query_emotion = self.get_emotion_vector(query_csv)
        if verbose:
            print(f"      Query emotion: {query_emotion}")
        
        # Get archive emotion vectors
        if verbose:
            print(f"\n[2/4] Getting archive emotion vectors from {archive_csv}")
        archive_emotions = {}
        for match in layer1_matches:
            movement_id = match['movement_id']
            archive_emotions[movement_id] = self.emotion_cache.get(
                archive_csv,
                self.get_emotion_vector(archive_csv)
            )
        if verbose:
            print(f"      Loaded {len(archive_emotions)} emotion vectors")
        
        # Rerank by emotion
        if verbose:
            print(f"\n[3/4] Reranking Layer 1 matches by emotion (alpha={self.alpha})")
        final_matches = self.rerank_layer1_matches(
            layer1_matches,
            query_emotion,
            archive_emotions,
            top_k=top_k
        )
        
        if verbose:
            num_shown = min(top_k, len(final_matches))
            print(f"      Reranked {len(layer1_matches)} matches -> Top {num_shown}")
        
        # Display results
        if verbose:
            print(f"\n[4/4] Final Results (alpha={self.alpha})")
            print("      " + "-"*76)
            print(f"      {'Rank':<5} {'Movement':<15} {'Body':<8} {'Emotion':<8} {'Final':<8}")
            print("      " + "-"*76)
            for i, match in enumerate(final_matches, 1):
                print(f"      {i:<5} {match['movement_id']:<15} "
                      f"{match['body_similarity']:<8.3f} "
                      f"{match['emotion_similarity']:<8.3f} "
                      f"{match['final_score']:<8.3f}")
            print("      " + "-"*76)
        
        return final_matches


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def example_usage():
    """Example: Complete Layer 1 + Layer 2 matching."""
    
    matcher = Layer1Layer2Matcher(
        layer2_model_path="saved_models/emotion_model_best.pt",
        layer2_metadata_path="saved_models/emotion_model_metadata.json",
        device='cpu',
        alpha=0.6
    )
    
    # Simulated Layer 1 results
    layer1_matches = [
        {"movement_id": "move_001", "body_similarity": 0.94},
        {"movement_id": "move_002", "body_similarity": 0.91},
        {"movement_id": "move_003", "body_similarity": 0.89},
        {"movement_id": "move_004", "body_similarity": 0.87},
        {"movement_id": "move_005", "body_similarity": 0.85},
        {"movement_id": "move_006", "body_similarity": 0.83},
        {"movement_id": "move_007", "body_similarity": 0.81},
        {"movement_id": "move_008", "body_similarity": 0.79},
        {"movement_id": "move_009", "body_similarity": 0.77},
        {"movement_id": "move_010", "body_similarity": 0.75},
    ]
    
    query_csv = "pose_results_L2/gBR_sBM_c01_d04_mBR0_ch01_pose.csv"
    archive_csv = "pose_results_L2/gHO_sBM_c01_d19_mHO0_ch01_pose.csv"
    
    final_matches = matcher.full_matching_pipeline(
        query_csv,
        layer1_matches,
        archive_csv,
        top_k=5,
        verbose=True
    )
    
    output_path = "pose_matching_output_final/layer1_layer2_integration.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(convert_to_serializable(final_matches), f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return final_matches


if __name__ == "__main__":
    example_usage()
