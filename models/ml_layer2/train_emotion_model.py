"""
STEP3: ML Models - Layer 2 (Emotional Similarity)
TRAINING UTILITY: Train Emotion Model (THIS FILE)

PURPOSE: Training script for the emotion/style classification model.
Processes extracted emotion features and trains the neural network.
Output: saved_models/emotion_model.pt - used in predict_emotion.py
"""

"""
Training pipeline for the emotion prediction model.

Supports both multi-task learning (regression + classification) and
single-task regression depending on available labels.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from typing import Dict, Tuple, Optional

from models.ml_layer2.extract_emotion_features import process_all_csvs
from models.ml_layer2.emotion_model import EmotionNet, EmotionNetSimple


class EmotionDataset(Dataset):
    """
    PyTorch Dataset for emotion prediction.
    Handles both regression and classification labels.
    """
    
    def __init__(self, X: np.ndarray, labels: Dict):
        """
        Args:
            X: Feature matrix (num_samples, num_features)
            labels: Dictionary containing regression and classification labels
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.labels = labels
        
        # Regression targets
        self.regression_targets = []
        self.regression_cols = labels.get('regression_targets', [])
        for col in self.regression_cols:
            if col in labels:
                self.regression_targets.append(torch.tensor(labels[col], dtype=torch.float32))
        
        # Classification targets
        self.classification_targets = {}
        self.classification_cols = labels.get('classification_targets', [])
        for col in self.classification_cols:
            if col in labels:
                self.classification_targets[col] = torch.tensor(labels[col], dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        
        # Pack all labels
        sample = {'x': x}
        
        # Regression labels
        for i, col in enumerate(self.regression_cols):
            if i < len(self.regression_targets):
                sample[f'reg_{col}'] = self.regression_targets[i][idx]
        
        # Classification labels
        for col in self.classification_cols:
            if col in self.classification_targets:
                sample[f'clf_{col}'] = self.classification_targets[col][idx]
        
        return sample


def create_dataloaders(
    X: np.ndarray,
    labels: Dict,
    batch_size: int = 16,
    train_split: float = 0.8,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        X: Feature matrix
        labels: Labels dictionary
        batch_size: Batch size
        train_split: Proportion of data for training
        seed: Random seed for reproducibility
    
    Returns:
        (train_dataloader, val_dataloader)
    """
    
    np.random.seed(seed)
    n = len(X)
    indices = np.random.permutation(n)
    train_idx = indices[:int(n * train_split)]
    val_idx = indices[int(n * train_split):]
    
    X_train, X_val = X[train_idx], X[val_idx]
    
    # Split labels
    labels_train = {}
    labels_val = {}
    
    for key, value in labels.items():
        if isinstance(value, np.ndarray):
            labels_train[key] = value[train_idx]
            labels_val[key] = value[val_idx]
        else:
            labels_train[key] = value
            labels_val[key] = value
    
    train_dataset = EmotionDataset(X_train, labels_train)
    val_dataset = EmotionDataset(X_val, labels_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    regression_weight: float = 0.7,
    classification_weight: float = 0.3
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training dataloader
        optimizer: Optimizer
        device: Device (CPU or GPU)
        regression_weight: Weight for regression loss in combined loss
        classification_weight: Weight for classification loss in combined loss
    
    Returns:
        Average loss for the epoch
    """
    
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Loss functions
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    for batch in train_loader:
        x = batch['x'].to(device)
        outputs = model(x)
        
        # Regression loss
        reg_loss = 0
        num_reg_targets = 0
        for key in batch.keys():
            if key.startswith('reg_'):
                y = batch[key].to(device).unsqueeze(1)
                pred = outputs['regression'][:, num_reg_targets:num_reg_targets+1]
                reg_loss += mse_loss(pred, y)
                num_reg_targets += 1
        
        if num_reg_targets > 0:
            reg_loss = reg_loss / num_reg_targets
        
        # Classification loss
        clf_loss = 0
        num_clf_targets = 0
        for key in batch.keys():
            if key.startswith('clf_'):
                y = batch[key].to(device)
                target_name = key.replace('clf_', '')
                if target_name == 'emotion_primary':
                    pred = outputs['emotion_primary_logits']
                elif target_name == 'memory_tag':
                    pred = outputs['memory_tag_logits']
                else:
                    continue
                clf_loss += ce_loss(pred, y)
                num_clf_targets += 1
        
        if num_clf_targets > 0:
            clf_loss = clf_loss / num_clf_targets
        
        # Combined loss
        if num_reg_targets > 0 and num_clf_targets > 0:
            loss = regression_weight * reg_loss + classification_weight * clf_loss
        elif num_reg_targets > 0:
            loss = reg_loss
        elif num_clf_targets > 0:
            loss = clf_loss
        else:
            continue
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    regression_weight: float = 0.7,
    classification_weight: float = 0.3
) -> float:
    """
    Validate the model.
    
    Args:
        model: Neural network model
        val_loader: Validation dataloader
        device: Device (CPU or GPU)
        regression_weight: Weight for regression loss
        classification_weight: Weight for classification loss
    
    Returns:
        Average validation loss
    """
    
    model.eval()
    total_loss = 0
    num_batches = 0
    
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch['x'].to(device)
            outputs = model(x)
            
            # Regression loss
            reg_loss = 0
            num_reg_targets = 0
            for key in batch.keys():
                if key.startswith('reg_'):
                    y = batch[key].to(device).unsqueeze(1)
                    pred = outputs['regression'][:, num_reg_targets:num_reg_targets+1]
                    reg_loss += mse_loss(pred, y)
                    num_reg_targets += 1
            
            if num_reg_targets > 0:
                reg_loss = reg_loss / num_reg_targets
            
            # Classification loss
            clf_loss = 0
            num_clf_targets = 0
            for key in batch.keys():
                if key.startswith('clf_'):
                    y = batch[key].to(device)
                    target_name = key.replace('clf_', '')
                    if target_name == 'emotion_primary':
                        pred = outputs['emotion_primary_logits']
                    elif target_name == 'memory_tag':
                        pred = outputs['memory_tag_logits']
                    else:
                        continue
                    clf_loss += ce_loss(pred, y)
                    num_clf_targets += 1
            
            if num_clf_targets > 0:
                clf_loss = clf_loss / num_clf_targets
            
            # Combined loss
            if num_reg_targets > 0 and num_clf_targets > 0:
                loss = regression_weight * reg_loss + classification_weight * clf_loss
            elif num_reg_targets > 0:
                loss = reg_loss
            elif num_clf_targets > 0:
                loss = clf_loss
            else:
                continue
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


def train_model(
    csv_dir: str = "../pose_results_L2",
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    output_dir: str = "../saved_models",
    use_gpu: bool = True,
    save_metadata: bool = True
):
    """
    Complete training pipeline.
    
    Args:
        csv_dir: Directory containing CSV files
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        output_dir: Directory to save model
        use_gpu: Whether to use GPU if available
        save_metadata: Whether to save training metadata
    """
    
    # Setup device
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and process data
    print("Loading and processing data...")
    X, labels = process_all_csvs(csv_dir)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(X, labels, batch_size=batch_size)
    
    # Create model
    print("Creating model...")
    input_dim = X.shape[1]
    
    emotion_primary_classes = labels.get('emotion_primary_classes', 5)
    memory_tag_classes = labels.get('memory_tag_classes', 10)
    
    model = EmotionNet(
        input_dim=input_dim,
        regression_output_dim=5,
        emotion_primary_classes=emotion_primary_classes,
        memory_tag_classes=memory_tag_classes
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"{output_dir}/emotion_model_best.pt")
            print(f"  -> Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Save final model
    torch.save(model.state_dict(), f"{output_dir}/emotion_model_final.pt")
    
    # Save metadata
    if save_metadata:
        metadata = {
            'input_dim': input_dim,
            'regression_output_dim': 5,
            'emotion_primary_classes': emotion_primary_classes,
            'memory_tag_classes': memory_tag_classes,
            'feature_names': labels.get('feature_names', []),
            'regression_targets': labels.get('regression_targets', []),
            'classification_targets': labels.get('classification_targets', []),
            'emotion_primary_mapping': labels.get('emotion_primary_mapping', {}),
            'memory_tag_mapping': labels.get('memory_tag_mapping', {}),
        }
        
        with open(f"{output_dir}/emotion_model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {output_dir}/emotion_model_metadata.json")
    
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to {output_dir}/emotion_model_best.pt")


if __name__ == "__main__":
    train_model()
