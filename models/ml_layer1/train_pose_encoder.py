"""
STEP3: ML Models - Layer 1 (Body Similarity)
TRAINING UTILITY: Train Pose Encoder (THIS FILE)

PURPOSE: Training script for the pose encoder model.
Processes pose CSV data and trains a neural network to generate discriminative embeddings.
Output: saved_models/pose_encoder.pt - the trained model used in pose_inference.py
"""

# ml_layer1/train_pose_encoder.py

import os
import glob
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from pathlib import Path

from pose_model import PoseEncoder


class PoseTripletDataset(Dataset):
    def __init__(self, csv_dir):
        """
        Load pose CSVs where each row is a single landmark.
        Each frame has 33 landmarks with (x, y, z) coordinates.
        We convert each frame into a flattened (x, y) vector per landmark.
        """
        csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {csv_dir}")
        
        print(f"Loading {len(csv_files)} CSV files...")
        
        # Collect all frames from all CSVs
        all_frames = []
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            
            # Group by frame_index to get all landmarks for a frame
            for frame_idx, frame_group in df.groupby('frame_index'):
                # Extract x, y coordinates and sort by landmark_id
                landmarks = frame_group.sort_values('landmark_id')[['landmark_id', 'x', 'y']].values
                
                # Create pose vector: [x0, y0, x1, y1, ..., x32, y32]
                pose = landmarks[:, 1:].flatten().astype(np.float32)
                
                # Only keep complete frames (33 landmarks)
                if len(landmarks) == 33 and len(pose) == 66:
                    all_frames.append(pose)
        
        self.poses = np.array(all_frames, dtype=np.float32)
        print(f"Loaded {len(self.poses)} complete frames with shape {self.poses.shape}")

    def __len__(self):
        return len(self.poses)

    def normalize_pose(self, pose):
        """Normalize pose to be size and translation invariant."""
        pose = pose.reshape(-1, 2)

        # Center pose around centroid
        center = np.mean(pose, axis=0)
        pose = pose - center

        # Scale pose so body size differences matter less
        scale = np.linalg.norm(pose)
        if scale > 0:
            pose = pose / scale

        return pose.flatten()

    def augment_pose(self, pose):
        """Add small noise to normalized pose to create positive sample."""
        pose = pose.reshape(-1, 2)

        # Small noise makes a positive sample
        noise = np.random.normal(0, 0.01, pose.shape)
        augmented = pose + noise

        return augmented.flatten()

    def __getitem__(self, idx):
        anchor = self.poses[idx]
        anchor = self.normalize_pose(anchor)

        # Augment the NORMALIZED anchor to create positive
        positive = self.augment_pose(anchor)

        negative_idx = random.randint(0, len(self.poses) - 1)
        while negative_idx == idx:
            negative_idx = random.randint(0, len(self.poses) - 1)

        negative = self.poses[negative_idx]
        negative = self.normalize_pose(negative)

        anchor = torch.tensor(anchor, dtype=torch.float32)
        positive = torch.tensor(positive, dtype=torch.float32)
        negative = torch.tensor(negative, dtype=torch.float32)

        return anchor, positive, negative


def train():
    # Set up paths
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    csv_dir = project_root / "pose_results"
    save_dir = script_dir / "saved_models"
    save_dir.mkdir(exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"Loading data from {csv_dir}")
    dataset = PoseTripletDataset(str(csv_dir))
    print(f"Loaded {len(dataset)} pose samples with dimension {dataset.poses.shape[1]}")
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = dataset.poses.shape[1]
    model = PoseEncoder(input_dim=input_dim, embedding_dim=32).to(device)

    criterion = nn.TripletMarginLoss(margin=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for anchor, positive, negative in dataloader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)

            loss = criterion(
                anchor_embedding,
                positive_embedding,
                negative_embedding
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        scheduler.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    torch.save(model.state_dict(), save_dir / "pose_encoder.pt")
    print(f"\nModel saved to {save_dir / 'pose_encoder.pt'}")


if __name__ == "__main__":
    train()