"""
Training Pipeline for Learned A* Heuristics with Admissibility Enforcement
===========================================================================
Trains HeuristicCNN to predict distance-to-goal maps from obstacle and goal configurations.
Enhanced with admissibility-aware loss to ensure predictions don't overestimate distances.

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
import torch.nn.functional as F

from dataset import PathPlanningDataset, create_dataloader
from learned_heuristic_encoder import HeuristicCNNV

class AdmissibleDistanceLoss(nn.Module):
    """
    Loss function that enforces admissibility by heavily penalizing overestimation.

    Admissibility means the heuristic never overestimates the true cost to goal.
    This is critical for A* optimality: overestimation can cause suboptimal paths,
    while underestimation only reduces efficiency but maintains optimality.

    Args:
        overestimation_penalty: Weight for penalizing inadmissible predictions (default: 10.0)
        loss_type: Base loss function ('mse' or 'l1')
    """

    def __init__(self, overestimation_penalty=10.0, loss_type='mse'):
        super().__init__()
        self.overestimation_penalty = overestimation_penalty
        self.loss_type = loss_type

    def forward(self, pred_distances, target_distances, obstacle_map):
        """
        Args:
            pred_distances: [B, 1, H, W] - Predicted distance map
            target_distances: [B, 1, H, W] - Ground truth distance map
            obstacle_map: [B, 1, H, W] - Binary map (1=passable, 0=obstacle)

        Returns:
            total_loss: Weighted combination of overestimation and underestimation
            over_loss: Overestimation component (for monitoring)
            under_loss: Underestimation component (for monitoring)
        """
        # Create mask for passable cells only
        mask = (obstacle_map > 0).float()

        # Compute raw errors (positive = overestimate, negative = underestimate)
        errors = pred_distances - target_distances

        # Separate overestimation (inadmissible) from underestimation (safe)
        overestimations = torch.relu(errors)    # Only positive errors
        underestimations = torch.relu(-errors)  # Only negative errors

        # Apply base loss function
        if self.loss_type == 'mse':
            over_element_loss = overestimations ** 2
            under_element_loss = underestimations ** 2
        elif self.loss_type == 'l1':
            over_element_loss = overestimations
            under_element_loss = underestimations
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Apply mask and compute mean over valid cells
        over_loss = (over_element_loss * mask).sum() / (mask.sum() + 1e-8)
        under_loss = (under_element_loss * mask).sum() / (mask.sum() + 1e-8)

        # Heavily penalize overestimation to encourage admissibility
        total_loss = self.overestimation_penalty * over_loss + under_loss

        return total_loss, over_loss, under_loss


class HeuristicTrainer:
    """
    Trainer class for HeuristicCNN model with admissibility enforcement.

    Handles training loop, validation, checkpointing, and comprehensive
    admissibility metric logging.
    """

    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 device='cuda',
                 learning_rate=1e-3,
                 weight_decay=1e-4,
                 overestimation_penalty=10.0,
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
            overestimation_penalty: Penalty weight for inadmissible predictions (default: 10.0)
            loss_type: Loss function type ('mse' or 'l1')
            checkpoint_dir: Directory to save model checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # Loss and optimizer (now using admissibility-aware loss)
        self.criterion = AdmissibleDistanceLoss(
            overestimation_penalty=overestimation_penalty,
            loss_type=loss_type
        )
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        ''' self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        ) '''
        #testing new scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[25, 50, 75],  # Reduce at these epochs
            gamma=0.5
        )

        # Tracking
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.current_epoch = 0

        # New: Admissibility tracking
        self.train_over_losses = []
        self.train_under_losses = []
        self.val_over_losses = []
        self.val_under_losses = []
        self.val_inadmissible_ratios = []
        self.val_avg_overestimations = []
        self.val_max_overestimations = []

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Trainer Initialized with Admissibility Enforcement")
        print(f"{'='*70}")
        print(f"Device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Optimizer: AdamW (lr={learning_rate}, weight_decay={weight_decay})")
        print(f"Loss: {loss_type.upper()} with overestimation penalty = {overestimation_penalty}x")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"{'='*70}\n")

    def train_epoch(self):
        """Run one training epoch with admissibility tracking."""
        self.model.train()
        epoch_loss = 0.0
        epoch_over_loss = 0.0
        epoch_under_loss = 0.0
        batch_count = 0

        for batch_idx, (obstacle_maps, goal_maps, target_distances) in enumerate(self.train_loader):
            # Move to device
            obstacle_maps = obstacle_maps.to(self.device)
            goal_maps = goal_maps.to(self.device)
            target_distances = target_distances.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            pred_distances = self.model(obstacle_maps, goal_maps)

            # Compute loss (returns 3 values now)
            loss, over_loss, under_loss = self.criterion(
                pred_distances, target_distances, obstacle_maps
            )

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            epoch_over_loss += over_loss.item()
            epoch_under_loss += under_loss.item()
            batch_count += 1

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_loss = epoch_loss / batch_count
                avg_over = epoch_over_loss / batch_count
                avg_under = epoch_under_loss / batch_count
                print(f"  Batch [{batch_idx+1}/{len(self.train_loader)}] - "
                      f"Loss: {loss.item():.6f} (Avg: {avg_loss:.6f}) | "
                      f"Over: {avg_over:.6f} | Under: {avg_under:.6f}")

        avg_epoch_loss = epoch_loss / batch_count
        avg_over_loss = epoch_over_loss / batch_count
        avg_under_loss = epoch_under_loss / batch_count

        self.train_losses.append(avg_epoch_loss)
        self.train_over_losses.append(avg_over_loss)
        self.train_under_losses.append(avg_under_loss)

        return avg_epoch_loss

    def validate(self):
        """Run validation with comprehensive admissibility analysis."""
        self.model.eval()
        val_loss = 0.0
        val_over_loss = 0.0
        val_under_loss = 0.0
        batch_count = 0

        # Admissibility tracking
        total_cells = 0
        inadmissible_cells = 0
        total_overestimation = 0.0
        max_overestimation = 0.0

        # Standard metrics
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
                loss, over_loss, under_loss = self.criterion(
                    pred_distances, target_distances, obstacle_maps
                )
                val_loss += loss.item()
                val_over_loss += over_loss.item()
                val_under_loss += under_loss.item()

                # Compute additional metrics (masked)
                mask = (obstacle_maps > 0).float()
                errors = torch.abs(pred_distances - target_distances) * mask
                total_mae += (errors.sum() / (mask.sum() + 1e-8)).item()
                total_max_error += errors.max().item()

                # Admissibility analysis
                overestimations = torch.relu(pred_distances - target_distances) * mask
                num_passable = mask.sum().item()
                num_inadmissible = (overestimations > 0).sum().item()

                total_cells += num_passable
                inadmissible_cells += num_inadmissible
                total_overestimation += overestimations.sum().item()
                max_overestimation = max(max_overestimation, overestimations.max().item())

                batch_count += 1

        avg_val_loss = val_loss / batch_count
        avg_over_loss = val_over_loss / batch_count
        avg_under_loss = val_under_loss / batch_count
        avg_mae = total_mae / batch_count
        avg_max_error = total_max_error / batch_count

        # Admissibility metrics
        inadmissible_ratio = inadmissible_cells / (total_cells + 1e-8)
        avg_overestimation = total_overestimation / (total_cells + 1e-8)

        self.val_losses.append(avg_val_loss)
        self.val_over_losses.append(avg_over_loss)
        self.val_under_losses.append(avg_under_loss)
        self.val_inadmissible_ratios.append(inadmissible_ratio)
        self.val_avg_overestimations.append(avg_overestimation)
        self.val_max_overestimations.append(max_overestimation)

        return (avg_val_loss, avg_over_loss, avg_under_loss,
                avg_mae, avg_max_error,
                inadmissible_ratio, avg_overestimation, max_overestimation)

    def save_checkpoint(self, filename='checkpoint.pth', is_best=False):
        """Save model checkpoint with admissibility metrics."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_over_losses': self.train_over_losses,
            'train_under_losses': self.train_under_losses,
            'val_over_losses': self.val_over_losses,
            'val_under_losses': self.val_under_losses,
            'val_inadmissible_ratios': self.val_inadmissible_ratios,
            'val_avg_overestimations': self.val_avg_overestimations,
            'val_max_overestimations': self.val_max_overestimations,
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

        # Load admissibility metrics if available
        if 'train_over_losses' in checkpoint:
            self.train_over_losses = checkpoint['train_over_losses']
            self.train_under_losses = checkpoint['train_under_losses']
            self.val_over_losses = checkpoint['val_over_losses']
            self.val_under_losses = checkpoint['val_under_losses']
            self.val_inadmissible_ratios = checkpoint['val_inadmissible_ratios']
            self.val_avg_overestimations = checkpoint['val_avg_overestimations']
            self.val_max_overestimations = checkpoint['val_max_overestimations']

        print(f"  ✓ Checkpoint loaded: {checkpoint_path} (Epoch {self.current_epoch})")
        return True

    def train(self, num_epochs, save_every=10, visualize_every=20):
        """
        Main training loop with admissibility monitoring.

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            visualize_every: Create visualization every N epochs
        """
        print(f"\n{'='*70}")
        print(f"Starting Training with Admissibility Enforcement")
        print(f"Overestimation Penalty: {self.criterion.overestimation_penalty}x")
        print(f"{'='*70}\n")

        start_time = time.time()

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Training
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            train_loss = self.train_epoch()

            # Validation
            (val_loss, val_over_loss, val_under_loss,
             val_mae, val_max_error,
             inadmissible_ratio, avg_overest, max_overest) = self.validate()

            # Learning rate scheduling
            #self.scheduler.step(val_loss)
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Timing
            epoch_time = time.time() - epoch_start_time

            # Print epoch summary with admissibility info
            print(f"\n  Epoch Summary:")
            print(f"    Train Loss: {train_loss:.6f}")
            print(f"    Val Loss:   {val_loss:.6f}")
            print(f"      ├─ Overestimation Loss:  {val_over_loss:.6f}")
            print(f"      └─ Underestimation Loss: {val_under_loss:.6f}")
            print(f"    Val MAE:    {val_mae:.4f}")
            print(f"    Val Max Err: {val_max_error:.4f}")
            print(f"    📊 ADMISSIBILITY METRICS:")
            print(f"      ├─ Inadmissible Cells: {inadmissible_ratio*100:.2f}% (Target: <1%)")
            print(f"      ├─ Avg Overestimation: {avg_overest:.4f}")
            print(f"      └─ Max Overestimation: {max_overest:.4f}")
            print(f"    Learning Rate: {current_lr:.2e}")
            print(f"    Time: {epoch_time:.2f}s")

            # Save best model (consider both loss and admissibility)
            if val_loss < self.best_val_loss and inadmissible_ratio < 0.02:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                print(f"  ⭐ Best validation loss improved by {improvement:.6f} "
                      f"(Inadmissible: {inadmissible_ratio*100:.2f}%)")
                self.save_checkpoint(is_best=True)

            # Periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(filename=f'checkpoint_epoch_{epoch+1}.pth')

            # Periodic visualization
            if (epoch + 1) % visualize_every == 0:
                self.visualize_predictions(epoch+1)
                self.visualize_admissibility(epoch+1)

            print(f"{'─'*70}\n")

        # Training complete
        total_time = time.time() - start_time
        final_inadm = self.val_inadmissible_ratios[-1] if self.val_inadmissible_ratios else 0

        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"{'='*70}")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Final inadmissible ratio: {final_inadm*100:.2f}%")
        if final_inadm < 0.01:
            print(f"✓ Excellent! Model is highly admissible (<1% violations)")
        elif final_inadm < 0.05:
            print(f"✓ Good! Model is mostly admissible (<5% violations)")
        else:
            print(f"⚠ Warning: Model has {final_inadm*100:.1f}% inadmissible predictions")
            print(f"  Consider increasing overestimation_penalty or training longer")
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
        print(f"  ✓ Prediction visualization saved: {viz_path}")

    def visualize_admissibility(self, epoch):
        """
        Visualize where the model overestimates (inadmissible regions).
        Red areas indicate cells where predicted distance > true distance.
        """
        self.model.eval()

        # Get one batch from validation set
        obstacle_maps, goal_maps, target_distances = next(iter(self.val_loader))
        obstacle_maps = obstacle_maps.to(self.device)
        goal_maps = goal_maps.to(self.device)
        target_distances = target_distances.to(self.device)

        # Generate predictions
        with torch.no_grad():
            pred_distances = self.model(obstacle_maps, goal_maps)

        # Visualize first 2 samples
        num_samples = min(2, obstacle_maps.size(0))
        fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            obstacle_map = obstacle_maps[i, 0].cpu().numpy()
            target_dist = target_distances[i, 0].cpu().numpy()
            pred_dist = pred_distances[i, 0].cpu().numpy()

            # Compute overestimation map
            errors = pred_dist - target_dist
            overestimations = np.maximum(0, errors)  # Only positive errors

            # Mask obstacles
            obstacle_mask = (obstacle_map == 0)
            target_dist_masked = np.copy(target_dist)
            pred_dist_masked = np.copy(pred_dist)
            overest_masked = np.copy(overestimations)

            target_dist_masked[obstacle_mask] = np.nan
            pred_dist_masked[obstacle_mask] = np.nan
            overest_masked[obstacle_mask] = np.nan

            # Plot 1: Obstacle map
            axes[i, 0].imshow(obstacle_map, cmap='gray')
            axes[i, 0].set_title('Obstacle Map', fontsize=10)
            axes[i, 0].axis('off')

            # Plot 2: Ground truth
            axes[i, 1].imshow(obstacle_map == 0, cmap='gray_r', vmin=0, vmax=1)
            im1 = axes[i, 1].imshow(target_dist_masked, cmap='viridis_r', alpha=0.9)
            axes[i, 1].set_title('Ground Truth', fontsize=10)
            axes[i, 1].axis('off')
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)

            # Plot 3: Predictions
            axes[i, 2].imshow(obstacle_map == 0, cmap='gray_r', vmin=0, vmax=1)
            im2 = axes[i, 2].imshow(pred_dist_masked, cmap='viridis_r', alpha=0.9)
            axes[i, 2].set_title('Predicted', fontsize=10)
            axes[i, 2].axis('off')
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)

            # Plot 4: Overestimation heatmap
            axes[i, 3].imshow(obstacle_map == 0, cmap='gray_r', vmin=0, vmax=1)
            im3 = axes[i, 3].imshow(overest_masked, cmap='Reds', alpha=0.9, vmin=0)

            # Count inadmissible cells
            inadm_count = (overestimations[~obstacle_mask] > 0).sum()
            total_count = (~obstacle_mask).sum()
            inadm_pct = 100 * inadm_count / total_count if total_count > 0 else 0

            title_color = 'green' if inadm_pct < 5 else ('orange' if inadm_pct < 10 else 'red')
            axes[i, 3].set_title(f'Overestimation Map\n{inadm_pct:.1f}% Inadmissible',
                                fontsize=10, color=title_color, fontweight='bold')
            axes[i, 3].axis('off')
            plt.colorbar(im3, ax=axes[i, 3], fraction=0.046, label='Overestimation')

        plt.suptitle(f'Admissibility Analysis - Epoch {epoch}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        viz_path = os.path.join(self.checkpoint_dir, f'admissibility_epoch_{epoch}.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Admissibility visualization saved: {viz_path}")

    def plot_training_curves(self):
        """Plot and save comprehensive training curves with admissibility metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        epochs = range(1, len(self.train_losses) + 1)

        # Plot 1: Overall loss
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)

        # Mark best validation
        if self.val_losses:
            best_epoch = np.argmin(self.val_losses) + 1
            axes[0, 0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5,
                              label=f'Best Val (Epoch {best_epoch})')

        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Total Loss', fontsize=12)
        axes[0, 0].set_title('Training Progress', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Loss components
        if self.train_over_losses and self.train_under_losses:
            axes[0, 1].plot(epochs, self.train_over_losses, 'r-',
                           label='Train Overestimation', linewidth=2, alpha=0.7)
            axes[0, 1].plot(epochs, self.train_under_losses, 'b-',
                           label='Train Underestimation', linewidth=2, alpha=0.7)
            axes[0, 1].plot(epochs, self.val_over_losses, 'r--',
                           label='Val Overestimation', linewidth=2)
            axes[0, 1].plot(epochs, self.val_under_losses, 'b--',
                           label='Val Underestimation', linewidth=2)
            axes[0, 1].set_xlabel('Epoch', fontsize=12)
            axes[0, 1].set_ylabel('Loss Component', fontsize=12)
            axes[0, 1].set_title('Loss Components', fontsize=14, fontweight='bold')
            axes[0, 1].legend(fontsize=9)
            axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Inadmissible ratio over time
        if self.val_inadmissible_ratios:
            axes[1, 0].plot(epochs, [r*100 for r in self.val_inadmissible_ratios],
                           'orange', linewidth=2, marker='o', markersize=3)
            axes[1, 0].axhline(y=1.0, color='g', linestyle='--', alpha=0.5,
                              label='1% Target (Excellent)', linewidth=2)
            axes[1, 0].axhline(y=5.0, color='orange', linestyle='--', alpha=0.5,
                              label='5% Threshold (Good)', linewidth=2)
            axes[1, 0].axhline(y=10.0, color='r', linestyle='--', alpha=0.5,
                              label='10% Limit', linewidth=2)
            axes[1, 0].set_xlabel('Epoch', fontsize=12)
            axes[1, 0].set_ylabel('Inadmissible Cells (%)', fontsize=12)
            axes[1, 0].set_title('Admissibility Violations', fontsize=14, fontweight='bold')
            axes[1, 0].legend(fontsize=9)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim(bottom=0)

        # Plot 4: Average overestimation
        if self.val_avg_overestimations:
            axes[1, 1].plot(epochs, self.val_avg_overestimations,
                           'purple', linewidth=2, marker='s', markersize=3)
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('Average Overestimation', fontsize=12)
            axes[1, 1].set_title('Mean Overestimation on Val Set', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(bottom=0)

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
          overestimation_penalty=50.0,# Was 10.0
          loss_type='mse',
          device='cuda',
          checkpoint_dir='./models',
          resume=None):
    """
    Main training function with admissibility enforcement.

    Args:
        npz_path: Path to NPZ dataset file
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        overestimation_penalty: Penalty weight for inadmissible predictions (default: 10.0)
                               Higher values = stronger admissibility enforcement
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

    # Create trainer with admissibility enforcement
    trainer = HeuristicTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        overestimation_penalty=overestimation_penalty,
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
    DATA_PATH = "bugtrap_forest_064_moore_c16.npz"
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-4  # Was 1e-3
    WEIGHT_DECAY = 1e-4
    OVERESTIMATION_PENALTY = 50  #was 10
    LOSS_TYPE = 'l1'  # Was 'mse'
    DEVICE = 'cuda'                # 'cuda' or 'cpu'
    CHECKPOINT_DIR = './models'
    RESUME_CHECKPOINT = None       # e.g., './models/checkpoint.pth' or None
    # ------------------------------------------

    print("\n" + "="*70)
    print("TRAINING HEURISTIC CNN WITH ADMISSIBILITY ENFORCEMENT")
    print("="*70)
    print(f"Dataset: {DATA_PATH}")
    print(f"Overestimation Penalty: {OVERESTIMATION_PENALTY}x")
    print(f"Target: <1% inadmissible cells (Excellent)")
    print(f"        <5% inadmissible cells (Good)")
    print("="*70 + "\n")

    # Run training
    train(
        npz_path=DATA_PATH,
        num_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        overestimation_penalty=OVERESTIMATION_PENALTY,
        loss_type=LOSS_TYPE,
        device=DEVICE,
        checkpoint_dir=CHECKPOINT_DIR,
        resume=RESUME_CHECKPOINT
    )