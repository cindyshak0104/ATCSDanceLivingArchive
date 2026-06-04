"""
STEP3: ML Models - Layer 2 (Emotional Similarity)
 3.2.1: Emotion Model Architecture
 3.2.2: Feature Extraction
 3.2.3: Emotion Prediction (THIS FILE)

PURPOSE: Inference module for emotion/style prediction.
Loads trained emotion model and predicts emotional content from pose sequences.
Integrates with STEP4 backend for real-time emotion-based filtering/reranking.
"""

"""
Inference module for emotion prediction.
Loads a trained model and predicts emotion vectors for input poses.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from models.ml_layer2.emotion_model import EmotionNet
from models.ml_layer2.extract_emotion_features import (
    load_and_process_csv,
    aggregate_by_frame,
    compute_motion_features,
    create_feature_label_pairs
)


class EmotionPredictor:
    """
    Wrapper class for emotion prediction.
    Handles model loading, preprocessing, and inference.
    """
    
    def __init__(
        self,
        model_path: str,
        metadata_path: str,
        device: str = 'cpu'
    ):
        """
        Args:
            model_path: Path to saved model weights (.pt file)
            metadata_path: Path to model metadata (.json file)
            device: Device to use ('cpu' or 'cuda')
        """
        
        self.device = torch.device(device)
        self.model_path = model_path
        self.metadata_path = metadata_path
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Create and load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Prepare mappings
        self.emotion_primary_mapping = self.metadata.get('emotion_primary_mapping', {})
        self.memory_tag_mapping = self.metadata.get('memory_tag_mapping', {})
        
        # Reverse mappings (for decoding predictions)
        self.emotion_primary_reverse = {v: k for k, v in self.emotion_primary_mapping.items()}
        self.memory_tag_reverse = {v: k for k, v in self.memory_tag_mapping.items()}
        
        self.feature_names = self.metadata.get('feature_names', [])
    
    def _load_model(self) -> EmotionNet:
        """Load model architecture and weights."""
        
        model = EmotionNet(
            input_dim=self.metadata['input_dim'],
            regression_output_dim=self.metadata['regression_output_dim'],
            emotion_primary_classes=self.metadata['emotion_primary_classes'],
            memory_tag_classes=self.metadata['memory_tag_classes']
        )
        
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        return model
    
    def predict(self, X: np.ndarray) -> Dict:
        """
        Predict emotion vectors and labels from features.
        
        Args:
            X: Feature matrix (num_samples, num_features)
        
        Returns:
            Dictionary containing:
            - 'emotion_vector': Emotion vectors (num_samples, 5)
            - 'emotion_primary': Predicted emotion categories
            - 'memory_tag': Predicted memory categories
        """
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
        
        # Extract outputs
        emotion_vectors = outputs['regression'].cpu().numpy()
        emotion_primary_logits = outputs['emotion_primary_logits'].cpu().numpy()
        memory_tag_logits = outputs['memory_tag_logits'].cpu().numpy()
        
        # Get predicted classes
        emotion_primary_ids = np.argmax(emotion_primary_logits, axis=1)
        memory_tag_ids = np.argmax(memory_tag_logits, axis=1)
        
        # Decode to labels
        emotion_primary_labels = [
            self.emotion_primary_reverse.get(id, f"unknown_{id}")
            for id in emotion_primary_ids
        ]
        memory_tag_labels = [
            self.memory_tag_reverse.get(id, f"unknown_{id}")
            for id in memory_tag_ids
        ]
        
        return {
            'emotion_vector': emotion_vectors,
            'emotion_primary': emotion_primary_labels,
            'emotion_primary_confidence': np.max(emotion_primary_logits, axis=1),
            'memory_tag': memory_tag_labels,
            'memory_tag_confidence': np.max(memory_tag_logits, axis=1),
        }
    
    def predict_from_csv(self, csv_path: str) -> Dict:
        """
        Predict emotion vectors directly from a CSV file.
        
        Args:
            csv_path: Path to pose CSV file
        
        Returns:
            Predictions and metadata
        """
        
        # Load and process CSV
        df = load_and_process_csv(csv_path)
        df = aggregate_by_frame(df)
        df = compute_motion_features(df)
        
        # Extract features
        X, _ = create_feature_label_pairs(df)
        
        # Predict
        predictions = self.predict(X)
        
        # Add movement IDs
        predictions['movement_ids'] = df['movement'].values
        predictions['video_names'] = df['video_name'].values
        predictions['frames'] = df['frame_index'].values
        
        return predictions
    
    def get_emotion_vector_for_movement(
        self,
        csv_path: str,
        movement_id: str = None
    ) -> Optional[np.ndarray]:
        """
        Get aggregated emotion vector for a specific movement.
        Averages emotion vectors across all frames of the movement.
        
        Args:
            csv_path: Path to pose CSV file
            movement_id: Movement ID to extract (if None, uses all)
        
        Returns:
            Single emotion vector or None if movement not found
        """
        
        predictions = self.predict_from_csv(csv_path)
        
        if movement_id is not None:
            mask = predictions['movement_ids'] == movement_id
            if not np.any(mask):
                return None
            emotion_vectors = predictions['emotion_vector'][mask]
        else:
            emotion_vectors = predictions['emotion_vector']
        
        # Average across frames
        aggregated = np.mean(emotion_vectors, axis=0)
        return aggregated


def batch_predict(
    csv_files: List[str],
    model_path: str,
    metadata_path: str,
    device: str = 'cpu'
) -> Dict:
    """
    Predict emotions for multiple CSV files.
    
    Args:
        csv_files: List of CSV file paths
        model_path: Path to model weights
        metadata_path: Path to metadata
        device: Device to use
    
    Returns:
        Combined predictions dictionary
    """
    
    predictor = EmotionPredictor(model_path, metadata_path, device=device)
    
    all_predictions = {
        'emotion_vector': [],
        'emotion_primary': [],
        'emotion_primary_confidence': [],
        'memory_tag': [],
        'memory_tag_confidence': [],
        'movement_ids': [],
        'video_names': [],
        'frames': [],
    }
    
    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        predictions = predictor.predict_from_csv(csv_file)
        
        for key in all_predictions.keys():
            if key in predictions:
                all_predictions[key].extend(predictions[key])
    
    # Convert lists to arrays
    for key in ['emotion_vector', 'emotion_primary_confidence', 'memory_tag_confidence']:
        all_predictions[key] = np.array(all_predictions[key])
    
    return all_predictions


if __name__ == "__main__":
    # Example usage
    model_path = "../saved_models/emotion_model_best.pt"
    metadata_path = "../saved_models/emotion_model_metadata.json"
    
    # Create predictor
    predictor = EmotionPredictor(model_path, metadata_path, device='cpu')
    
    # Predict from CSV
    csv_path = "../pose_results_L2/gBR_sBM_c01_d04_mBR0_ch01_pose.csv"
    predictions = predictor.predict_from_csv(csv_path)
    
    print("Prediction results:")
    print(f"Emotion vectors shape: {predictions['emotion_vector'].shape}")
    print(f"Sample emotion vector: {predictions['emotion_vector'][0]}")
    print(f"Sample emotion_primary: {predictions['emotion_primary'][0]}")
    print(f"Sample memory_tag: {predictions['memory_tag'][0]}")
