"""
STEP3: ML Models - Layer 2 (Emotional Similarity)
 3.2.1: Emotion Model Architecture (THIS FILE)
 3.2.2: Feature extraction from pose sequences
 3.2.3: Emotion prediction and reranking

PURPOSE: Defines the neural network for emotional/style classification.
Analyzes pose sequences to predict emotional content or dance style.
Used to rerank and filter pose matches based on emotional similarity.
"""

"""
Multi-task neural network for emotion prediction.

This model performs both regression and classification tasks:
- Regression: Predict continuous emotion values (emotion_intensity, arousal, tension, weight, flow)
- Classification: Predict categorical labels (emotion_primary, memory_tag)
"""

import torch
import torch.nn as nn


class EmotionNet(nn.Module):
    """
    Multi-task emotion prediction network.
    
    Architecture:
    - Shared encoder layers (all tasks benefit from pose features)
    - Task-specific heads:
      - Regression head for continuous emotion values
      - Classification heads for categorical emotion/memory labels
    """
    
    def __init__(
        self,
        input_dim: int,
        regression_output_dim: int = 5,
        emotion_primary_classes: int = 5,
        memory_tag_classes: int = 10
    ):
        """
        Args:
            input_dim: Number of input features (pose-derived)
            regression_output_dim: Number of regression outputs (emotion_intensity, arousal, tension, weight, flow)
            emotion_primary_classes: Number of emotion_primary categories
            memory_tag_classes: Number of memory_tag categories
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.regression_output_dim = regression_output_dim
        self.emotion_primary_classes = emotion_primary_classes
        self.memory_tag_classes = memory_tag_classes
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Regression head: predict continuous emotion values
        self.regression_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, regression_output_dim),
            nn.Sigmoid()  # Normalize to [0, 1] for emotion values
        )
        
        # Classification head: emotion_primary
        self.emotion_primary_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, emotion_primary_classes)
        )
        
        # Classification head: memory_tag
        self.memory_tag_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, memory_tag_classes)
        )
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Dictionary containing:
            - 'regression': Regression outputs (batch_size, regression_output_dim)
            - 'emotion_primary_logits': Classification logits for emotion_primary
            - 'memory_tag_logits': Classification logits for memory_tag
        """
        
        # Shared encoder
        encoded = self.encoder(x)
        
        # Task-specific outputs
        regression_output = self.regression_head(encoded)
        emotion_primary_logits = self.emotion_primary_head(encoded)
        memory_tag_logits = self.memory_tag_head(encoded)
        
        return {
            'regression': regression_output,
            'emotion_primary_logits': emotion_primary_logits,
            'memory_tag_logits': memory_tag_logits
        }
    
    def get_emotion_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the emotion vector (regression output) for a given input.
        This is used for similarity comparisons in Layer 1 + Layer 2 matching.
        
        Args:
            x: Input tensor
        
        Returns:
            Emotion vector of shape (batch_size, regression_output_dim)
        """
        outputs = self.forward(x)
        return outputs['regression']


class EmotionNetSimple(nn.Module):
    """
    Simplified single-task version for pure regression (emotion vector prediction).
    Use this if you want to focus only on continuous emotion values.
    """
    
    def __init__(self, input_dim: int, output_dim: int = 5):
        """
        Args:
            input_dim: Number of input features
            output_dim: Number of regression outputs
        """
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, output_dim),
            nn.Sigmoid()  # Normalize to [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returns emotion vector."""
        return self.model(x)


if __name__ == "__main__":
    # Example usage
    batch_size = 8
    input_dim = 25
    
    # Create model
    model = EmotionNet(
        input_dim=input_dim,
        regression_output_dim=5,
        emotion_primary_classes=5,
        memory_tag_classes=10
    )
    
    # Forward pass
    x = torch.randn(batch_size, input_dim)
    outputs = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Regression output shape: {outputs['regression'].shape}")
    print(f"Emotion primary logits shape: {outputs['emotion_primary_logits'].shape}")
    print(f"Memory tag logits shape: {outputs['memory_tag_logits'].shape}")
    
    # Get emotion vector
    emotion_vector = model.get_emotion_vector(x)
    print(f"Emotion vector shape: {emotion_vector.shape}")
