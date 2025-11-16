"""
Test Trained Heuristic Model
=============================
Load a trained model and visualize predictions on a specific map from the dataset.

Usage:
    python test_model.py --model_path models/best_heuristic_model.pth --data_path data.npz --index 0
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dataset import PathPlanningDataset
from learned_heuristic_encoder import HeuristicCNN


def test_model(model_path, data_path, map_index=0, split='test', save_path='model_test_result.png'):
    """
    Test trained model on a specific map and visualize results.
    
    Args:
        model_path: Path to trained model checkpoint (.pth file)
        data_path: Path to NPZ dataset
        map_index: Index of map to test on
        split: Dataset split ('train', 'valid', or 'test')
        save_path: Where to save visualization
    """
    
    print("\n" + "="*70)
    print("TESTING TRAINED HEURISTIC MODEL")
    print("="*70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load dataset
    print(f"\nLoading dataset from: {data_path}")
    dataset = PathPlanningDataset(data_path, split=split)
    
    if map_index >= len(dataset):
        print(f"✗ Error: Index {map_index} out of range (dataset has {len(dataset)} samples)")
        return
    
    # Get specific map
    obstacle_map, goal_map, target_distances = dataset[map_index]
    
    # Convert to tensors and add batch dimension
    obstacle_map = torch.from_numpy(obstacle_map).unsqueeze(0).to(device)  # [1, 1, H, W]
    goal_map = torch.from_numpy(goal_map).unsqueeze(0).to(device)          # [1, 1, H, W]
    target_distances = torch.from_numpy(target_distances).unsqueeze(0).to(device)  # [1, 1, H, W]
    
    print(f"Testing on map index: {map_index}")
    print(f"Map size: {obstacle_map.shape[2]}x{obstacle_map.shape[3]}")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = HeuristicCNN(in_channels=2, base_channels=64).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get model info from checkpoint
    if 'epoch' in checkpoint:
        print(f"Model trained for {checkpoint['epoch']} epochs")
    if 'best_val_loss' in checkpoint:
        print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
    
    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        pred_distances = model(obstacle_map, goal_map)
    
    # Move to CPU for visualization
    obstacle_map = obstacle_map.squeeze(0).cpu().numpy()  # [1, H, W]
    goal_map = goal_map.squeeze(0).cpu().numpy()
    target_distances = target_distances.squeeze(0).cpu().numpy()
    pred_distances = pred_distances.squeeze(0).cpu().numpy()
    
    # Remove channel dimension
    obstacle_map = obstacle_map[0]
    goal_map = goal_map[0]
    target_distances = target_distances[0]
    pred_distances = pred_distances[0]
    
    # Compute error metrics
    obstacle_mask = (obstacle_map > 0)
    valid_errors = np.abs(pred_distances - target_distances)[obstacle_mask]
    
    mae = np.mean(valid_errors)
    max_error = np.max(valid_errors)
    rmse = np.sqrt(np.mean(valid_errors ** 2))
    
    print(f"\nPrediction Metrics (on passable cells only):")
    print(f"  MAE (Mean Absolute Error): {mae:.4f}")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"  Max Error: {max_error:.4f}")
    print(f"  Ground truth range: [{target_distances[obstacle_mask].min():.2f}, {target_distances[obstacle_mask].max():.2f}]")
    print(f"  Predicted range: [{pred_distances[obstacle_mask].min():.2f}, {pred_distances[obstacle_mask].max():.2f}]")
    
    # ========== VISUALIZATION ==========
    print(f"\nCreating visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Find goal coordinates
    goal_coords = np.where(goal_map == 1)
    goal_y, goal_x = (goal_coords[0][0], goal_coords[1][0]) if len(goal_coords[0]) > 0 else (None, None)

    # Mask obstacles for heatmaps
    target_masked = np.copy(target_distances)
    pred_masked = np.copy(pred_distances)

    obstacle_vis_mask = (obstacle_map == 0)
    target_masked[obstacle_vis_mask] = np.nan
    pred_masked[obstacle_vis_mask] = np.nan

    # Column 1: Ground Truth
    axes[0].imshow(obstacle_vis_mask, cmap='gray_r', vmin=0, vmax=1)
    im1 = axes[0].imshow(target_masked, cmap='viridis_r', alpha=0.9, interpolation='nearest')
    if goal_x is not None:
        axes[0].scatter(goal_x, goal_y, marker='x', c='red', s=200, 
                        linewidths=3, zorder=10)
    axes[0].set_title('GROUND TRUTH: Optimal Distances', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    divider = make_axes_locatable(axes[0])
    cax1 = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im1, cax=cax1, label="Distance")

    # Column 2: Model Output (Predicted)
    axes[1].imshow(obstacle_vis_mask, cmap='gray_r', vmin=0, vmax=1)
    im2 = axes[1].imshow(pred_masked, cmap='viridis_r', alpha=0.9, interpolation='nearest')
    if goal_x is not None:
        axes[1].scatter(goal_x, goal_y, marker='x', c='red', s=200, 
                        linewidths=3, zorder=10)
    axes[1].set_title(f'MODEL OUTPUT: Predicted Distances\nMAE: {mae:.3f}', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    divider = make_axes_locatable(axes[1])
    cax2 = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im2, cax=cax2, label="Distance")

    # Overall title
    fig.suptitle(f'Model Test: Map {map_index} ({split.upper()} set)', 
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {save_path}")

    print("\n" + "="*70)
    print("TEST COMPLETE ✓")
    print("="*70 + "\n")

    plt.show()


if __name__ == "__main__":
    # Configuration variables - EDIT THESE
    model_path = 'models/best_heuristic_model.pth'
    data_path = r'E:\Project_CDJ\CDJ\04 - AREAS\UNIVERSITY\PURDUE\SCHOOL\FALL 2025\Artificial Intelligence ECE 57000\CLASS\PROJECTS\astar-efficient-ai\data\bugtrap_forest_064_moore_c16.npz'
    map_index = 0
    split = 'test'  # 'train', 'valid', or 'test'
    save_path = 'model_test_result.png'
    
    # Run test
    test_model(
        model_path=model_path,
        data_path=data_path,
        map_index=map_index,
        split=split,
        save_path=save_path
    )