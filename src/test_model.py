"""
Test Trained Heuristic Model
Load a trained model and visualize predictions on a specific map from the dataset.
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#from dataset import PathPlanningDataset
#from learned_heuristic_encoder import HeuristicCNN


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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    try:
        dataset = PathPlanningDataset(data_path, split=split)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if map_index >= len(dataset):
        print(f"Error: Index {map_index} out of range (dataset has {len(dataset)} samples)")
        return
    
    obstacle_map, goal_map, target_distances = dataset[map_index]
    
    # Prep tensors for model
    obstacle_map_t = torch.from_numpy(obstacle_map).unsqueeze(0).to(device)
    goal_map_t = torch.from_numpy(goal_map).unsqueeze(0).to(device)
    target_distances_t = torch.from_numpy(target_distances).unsqueeze(0).to(device)
    
    # Load model
    model = HeuristicCNN(in_channels=2, base_channels=64).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run inference
    with torch.no_grad():
        pred_distances_t = model(obstacle_map_t, goal_map_t)
    
    # Convert to numpy for visualization
    obstacle_map_np = obstacle_map[0]
    goal_map_np = goal_map[0]
    target_distances_np = target_distances[0]
    pred_distances_np = pred_distances_t.squeeze(0).squeeze(0).cpu().numpy()
    
    # Compute metrics on passable cells only
    obstacle_mask = (obstacle_map_np > 0)
    total_passable_cells = np.sum(obstacle_mask)
    
    # General error metrics
    abs_error = np.abs(pred_distances_np - target_distances_np)
    valid_errors = abs_error[obstacle_mask]
    mae = np.mean(valid_errors)
    rmse = np.sqrt(np.mean(valid_errors ** 2))
    
    # Overestimation (inadmissibility) metrics
    overestimation_error = np.maximum(0, pred_distances_np - target_distances_np)
    overestimation_on_passable = overestimation_error[obstacle_mask]
    num_inadmissible_cells = np.sum(overestimation_on_passable > 1e-5)
    inadmissible_ratio = num_inadmissible_cells / total_passable_cells if total_passable_cells > 0 else 0
    max_overestimation = np.max(overestimation_on_passable) if num_inadmissible_cells > 0 else 0
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Find goal location
    goal_coords = np.where(goal_map_np == 1)
    goal_y, goal_x = (goal_coords[0][0], goal_coords[1][0]) if len(goal_coords[0]) > 0 else (None, None)
    
    # Prepare masked arrays for visualization
    target_masked = np.copy(target_distances_np)
    pred_masked = np.copy(pred_distances_np)
    error_masked = np.copy(abs_error)
    overestimation_masked = np.copy(overestimation_error)
    
    obstacle_vis_mask = (obstacle_map_np == 0)
    target_masked[obstacle_vis_mask] = np.nan
    pred_masked[obstacle_vis_mask] = np.nan
    error_masked[obstacle_vis_mask] = np.nan
    overestimation_masked[obstacle_vis_mask] = np.nan
    
    # Top-left: Ground Truth
    axes[0, 0].imshow(obstacle_vis_mask, cmap='gray_r', vmin=0, vmax=1)
    im1 = axes[0, 0].imshow(target_masked, cmap='viridis_r', alpha=0.9, interpolation='nearest')
    if goal_x is not None:
        axes[0, 0].scatter(goal_x, goal_y, marker='x', c='red', s=200, linewidths=3, zorder=10)
    axes[0, 0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    divider1 = make_axes_locatable(axes[0, 0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im1, cax=cax1, label="Distance")
    
    # Top-right: Model Prediction
    axes[0, 1].imshow(obstacle_vis_mask, cmap='gray_r', vmin=0, vmax=1)
    im2 = axes[0, 1].imshow(pred_masked, cmap='viridis_r', alpha=0.9, interpolation='nearest')
    if goal_x is not None:
        axes[0, 1].scatter(goal_x, goal_y, marker='x', c='red', s=200, linewidths=3, zorder=10)
    axes[0, 1].set_title(f'Prediction (MAE: {mae:.3f})', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    divider2 = make_axes_locatable(axes[0, 1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im2, cax=cax2, label="Distance")
    
    # Bottom-left: Absolute Error (orange)
    axes[1, 0].imshow(obstacle_vis_mask, cmap='gray_r', vmin=0, vmax=1)
    im3 = axes[1, 0].imshow(error_masked, cmap='Oranges', alpha=0.9, interpolation='nearest', vmin=0)
    if goal_x is not None:
        axes[1, 0].scatter(goal_x, goal_y, marker='x', c='blue', s=200, linewidths=3, zorder=10)
    axes[1, 0].set_title(f'Absolute Error (RMSE: {rmse:.3f})', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    divider3 = make_axes_locatable(axes[1, 0])
    cax3 = divider3.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im3, cax=cax3, label="Error")
    
    # Bottom-right: Overestimation (red)
    axes[1, 1].imshow(obstacle_vis_mask, cmap='gray_r', vmin=0, vmax=1)
    im4 = axes[1, 1].imshow(overestimation_masked, cmap='Reds', alpha=0.9, interpolation='nearest', vmin=0)
    if goal_x is not None:
        axes[1, 1].scatter(goal_x, goal_y, marker='x', c='blue', s=200, linewidths=3, zorder=10)
    
    title_color = 'red' if inadmissible_ratio > 0.01 else 'green'
    axes[1, 1].set_title(f'Overestimation ({inadmissible_ratio*100:.2f}% cells)', 
                         fontsize=14, fontweight='bold', color=title_color)
    axes[1, 1].axis('off')
    divider4 = make_axes_locatable(axes[1, 1])
    cax4 = divider4.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im4, cax=cax4, label="Overestimation")
    
    fig.suptitle(f'Model Test: Map {map_index} ({split.upper()} set)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    model_path = 'models/best_heuristic_model.pth'
    data_path = "bugtrap_forest_064_moore_c16.npz"
    map_index = 0
    split = 'test'
    save_path = 'model_test_result1.png'
    save_path2 = 'model_test_result2.png'
    test_model(
        model_path=model_path,
        data_path=data_path,
        map_index=map_index,
        split=split,
        save_path=save_path
    )

    test_model(
        model_path=model_path,
        data_path=data_path,
        map_index=map_index + 1,
        split=split,
        save_path=save_path2
    )