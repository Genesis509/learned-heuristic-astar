"""
Training Pipeline for Learned A* Heuristics
===========================================
Trains HeuristicCNN to predict distance-to-goal maps from obstacle and goal configurations.

Usage:
    python train_heuristic.py --data_path <path_to_npz> --epochs 100 --batch_size 16
"""

import os
import argparse
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import PathPlanningDataset, create_dataloader
from learned_heuristic_encoder import HeuristicCNN


class MaskedDistanceLoss(nn.Module):
    """
    Loss function for distance prediction that ignores obstacle cells.
    
    Computes MSE only on passable cells where the model should predict distances.
    Obstacle cells (obstacle_map == 0) are masked out from loss computation.
    """
    
    def __init__(self, loss_type='mse'):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(self, pred_distances, target_distances, obstacle_map):
        """
        Args:
            pred_distances: [B, 1, H, W] - Predicted distance map
            target_distances: [B, 1, H, W] - Ground truth distance map
            obstacle_map: [B, 1, H, W] - Binary map (1=passable, 0=obstacle)
        
        Returns:
            loss: Scalar loss value
        """
        # Create mask for passable cells only
        mask = (obstacle_map > 0).float()
        
        # Compute element-wise loss
        if self.loss_type == 'mse':
            element_loss = (pred_distances - target_distances) ** 2
        elif self.loss_type == 'l1':
            element_loss = torch.abs(pred_distances - target_distances)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Apply mask and compute mean over valid cells
        masked_loss = element_loss * mask
        loss = masked_loss.sum() / (mask.sum() + 1e-8)  # Avoid division by zero
        
        return loss


class HeuristicTrainer:
    """
    Trainer class for HeuristicCNN model.
    
    Handles training loop, validation, checkpointing, and metric logging.
    """
    
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 device='cuda',
                 learning_rate=1e-3,
                 weight_decay=1e-4,
                 loss_type='mse',
                 checkpoint_dir='./models'):
        """
        Initialize trainer.
        
        Args:
            model: HeuristicCNN model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            loss_type: Loss function type ('mse' or 'l1')
            checkpoint_dir: Directory to save model checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Loss and optimizer
        self.criterion = MaskedDistanceLoss(loss_type=loss_type)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True,
            min_lr=1e-6
        )
        
        # Tracking
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.current_epoch = 0
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"Trainer Initialized")
        print(f"{'='*70}")
        print(f"Device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Optimizer: AdamW (lr={learning_rate}, weight_decay={weight_decay})")
        print(f"Loss: {loss_type.upper()}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"{'='*70}\n")
    
    def train_epoch(self):
        """Run one training epoch."""
        self.model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, (obstacle_maps, goal_maps, target_distances) in enumerate(self.train_loader):
            # Move to device
            obstacle_maps = obstacle_maps.to(self.device)
            goal_maps = goal_maps.to(self.device)
            target_distances = target_distances.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_distances = self.model(obstacle_maps, goal_maps)
            
            # Compute loss (masked to ignore obstacles)
            loss = self.criterion(pred_distances, target_distances, obstacle_maps)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            batch_count += 1
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_loss = epoch_loss / batch_count
                print(f"  Batch [{batch_idx+1}/{len(self.train_loader)}] - Loss: {loss.item():.6f} (Avg: {avg_loss:.6f})")
        
        avg_epoch_loss = epoch_loss / batch_count
        self.train_losses.append(avg_epoch_loss)
        return avg_epoch_loss
    
    def validate(self):
        """Run validation."""
        self.model.eval()
        val_loss = 0.0
        batch_count = 0
        
        # Metrics
        total_mae = 0.0
        total_max_error = 0.0
        
        with torch.no_grad():
            for obstacle_maps, goal_maps, target_distances in self.val_loader:
                # Move to device
                obstacle_maps = obstacle_maps.to(self.device)
                goal_maps = goal_maps.to(self.device)
                target_distances = target_distances.to(self.device)
                
                # Forward pass
                pred_distances = self.model(obstacle_maps, goal_maps)
                
                # Compute loss
                loss = self.criterion(pred_distances, target_distances, obstacle_maps)
                val_loss += loss.item()
                
                # Compute additional metrics (masked)
                mask = (obstacle_maps > 0).float()
                errors = torch.abs(pred_distances - target_distances) * mask
                total_mae += (errors.sum() / (mask.sum() + 1e-8)).item()
                total_max_error += errors.max().item()
                
                batch_count += 1
        
        avg_val_loss = val_loss / batch_count
        avg_mae = total_mae / batch_count
        avg_max_error = total_max_error / batch_count
        
        self.val_losses.append(avg_val_loss)
        
        return avg_val_loss, avg_mae, avg_max_error
    
    def save_checkpoint(self, filename='checkpoint.pth', is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_heuristic_model.pth')
            torch.save(checkpoint, best_path)
            print(f"  ✓ New best model saved: {best_path}")
    
    def load_checkpoint(self, filename='checkpoint.pth'):
        """Load model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(checkpoint_path):
            print(f"  ✗ Checkpoint not found: {checkpoint_path}")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"  ✓ Checkpoint loaded: {checkpoint_path} (Epoch {self.current_epoch})")
        return True
    
    def train(self, num_epochs, save_every=10, visualize_every=20):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            visualize_every: Create visualization every N epochs
        """
        print(f"\n{'='*70}")
        print(f"Starting Training")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            train_loss = self.train_epoch()
            
            # Validation
            val_loss, val_mae, val_max_error = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Timing
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            print(f"\n  Epoch Summary:")
            print(f"    Train Loss: {train_loss:.6f}")
            print(f"    Val Loss:   {val_loss:.6f}")
            print(f"    Val MAE:    {val_mae:.4f}")
            print(f"    Val Max Err: {val_max_error:.4f}")
            print(f"    Learning Rate: {current_lr:.2e}")
            print(f"    Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                print(f"  ⭐ Best validation loss improved by {improvement:.6f}")
                self.save_checkpoint(is_best=True)
            
            # Periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(filename=f'checkpoint_epoch_{epoch+1}.pth')
            
            # Periodic visualization
            if (epoch + 1) % visualize_every == 0:
                self.visualize_predictions(epoch+1)
            
            print(f"{'─'*70}\n")
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"{'='*70}")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Final model saved to: {self.checkpoint_dir}")
        print(f"{'='*70}\n")
        
        # Plot training curves
        self.plot_training_curves()
    
    def visualize_predictions(self, epoch):
        """Visualize model predictions on validation set."""
        self.model.eval()
        
        # Get one batch from validation set
        obstacle_maps, goal_maps, target_distances = next(iter(self.val_loader))
        obstacle_maps = obstacle_maps.to(self.device)
        goal_maps = goal_maps.to(self.device)
        target_distances = target_distances.to(self.device)
        
        # Generate predictions
        with torch.no_grad():
            pred_distances = self.model(obstacle_maps, goal_maps)
        
        # Visualize first 4 samples
        num_samples = min(4, obstacle_maps.size(0))
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            obstacle_map = obstacle_maps[i, 0].cpu().numpy()
            target_dist = target_distances[i, 0].cpu().numpy()
            pred_dist = pred_distances[i, 0].cpu().numpy()
            
            # Mask obstacles
            obstacle_mask = (obstacle_map == 0)
            target_dist_masked = np.copy(target_dist)
            pred_dist_masked = np.copy(pred_dist)
            target_dist_masked[obstacle_mask] = np.nan
            pred_dist_masked[obstacle_mask] = np.nan
            
            # Plot obstacle map
            axes[i, 0].imshow(obstacle_map, cmap='gray')
            axes[i, 0].set_title('Obstacle Map', fontsize=10)
            axes[i, 0].axis('off')
            
            # Plot ground truth
            axes[i, 1].imshow(obstacle_map == 0, cmap='gray_r', vmin=0, vmax=1)
            im1 = axes[i, 1].imshow(target_dist_masked, cmap='viridis_r', alpha=0.9)
            axes[i, 1].set_title('Ground Truth Distances', fontsize=10)
            axes[i, 1].axis('off')
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
            
            # Plot predictions
            axes[i, 2].imshow(obstacle_map == 0, cmap='gray_r', vmin=0, vmax=1)
            im2 = axes[i, 2].imshow(pred_dist_masked, cmap='viridis_r', alpha=0.9)
            axes[i, 2].set_title('Predicted Distances', fontsize=10)
            axes[i, 2].axis('off')
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
        
        plt.tight_layout()
        viz_path = os.path.join(self.checkpoint_dir, f'predictions_epoch_{epoch}.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Visualization saved: {viz_path}")
    
    def plot_training_curves(self):
        """Plot and save training/validation loss curves."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        ax.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        # Mark best validation
        best_epoch = np.argmin(self.val_losses) + 1
        ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best Val (Epoch {best_epoch})')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        curve_path = os.path.join(self.checkpoint_dir, 'training_curves.png')
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Training curves saved: {curve_path}")


def train(npz_path,
          num_epochs=100,
          batch_size=16,
          learning_rate=1e-3,
          weight_decay=1e-4,
          loss_type='mse',
          device='cuda',
          checkpoint_dir='./models',
          resume=None):
    """
    Main training function.
    
    Args:
        npz_path: Path to NPZ dataset file
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        loss_type: Loss function type ('mse' or 'l1')
        device: Device to train on ('cuda' or 'cpu')
        checkpoint_dir: Directory to save checkpoints
        resume: Path to checkpoint to resume from (optional)
    """
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Create dataloaders
    print("Loading datasets...")
    train_loader = create_dataloader(npz_path, split='train', batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(npz_path, split='valid', batch_size=batch_size, shuffle=False)
    
    # Create model
    print("Initializing model...")
    model = HeuristicCNN(in_channels=2, base_channels=64)
    
    # Create trainer
    trainer = HeuristicTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        loss_type=loss_type,
        checkpoint_dir=checkpoint_dir
    )
    
    # Resume from checkpoint if specified
    if resume:
        trainer.load_checkpoint(resume)
    
    # Train
    trainer.train(num_epochs=num_epochs)
    
    return trainer


if __name__ == "__main__":
    
    # --- Define your training settings here ---
    DATA_PATH = r"data/bugtrap_forest_064_moore_c16.npz" # Use the relative path from your project root
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    LOSS_TYPE = 'mse'      # 'mse' or 'l1'
    DEVICE = 'cuda'        # 'cuda' or 'cpu'
    CHECKPOINT_DIR = './models'
    RESUME_CHECKPOINT = None # e.g., './models/checkpoint.pth' or None
    # ------------------------------------------

    # You also need to import argparse to avoid an error,
    # or remove it from the top 'import' list.
    # We just keep it simple and call train() directly.
    
    print("--- Running with hardcoded settings ---")
    
    # Run training
    train(
        npz_path=DATA_PATH,
        num_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        loss_type=LOSS_TYPE,
        device=DEVICE,
        checkpoint_dir=CHECKPOINT_DIR,
        resume=RESUME_CHECKPOINT
    )