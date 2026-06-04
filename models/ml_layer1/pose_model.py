"""
STEP3: ML Models - Layer 1 (Body Similarity)
 3.1.1: Pose Matching
 3.1.2: Pose Inference / Feature Extraction
 3.1.3: Neural Network Architecture (THIS FILE)

PURPOSE: Defines the pose encoder neural network architecture.
Simple multi-layer perceptron that learns to embed 33-point poses into fixed-size vectors.
Trained via train_pose_encoder.py and used by pose_inference.py for inference.
"""

# ml_layer1/pose_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseEncoder(nn.Module):
    def __init__(self, input_dim=66, embedding_dim=32):
        super(PoseEncoder, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, embedding_dim)
        )

    def forward(self, x):
        embedding = self.network(x)

        # Normalize embedding so cosine similarity works better
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding