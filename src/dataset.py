"""
Dataset Loader for Modified A* Path Planning
Loads maze environments with ground truth optimal distances for supervised learning
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from mpl_toolkits.axes_grid1 import make_axes_locatable

class PathPlanningDataset(Dataset):
    """
    Dataset for path planning with optimal distance supervision.
    
    Loads NPZ files containing:
    - map_designs: Obstacle maps (1 = passable, 0 = obstacle)
    - goal_maps: One-hot encoded goal locations
    - opt_dists: Optimal distances from each cell to goal
    
    Output format for training encoder:
    - obstacle_map: [1, 64, 64] - Binary obstacle map
    - goal_map: [1, 64, 64] - One-hot goal location
    - optimal_distances: [1, 64, 64] - Ground truth costs (supervision signal)
    """
    
    def __init__(self, npz_path: str, split: str = 'train'):
        """
        Initialize dataset from NPZ file.
        
        Args:
            npz_path: Path to .npz file containing dataset
            split: One of 'train', 'valid', or 'test'
        """
        assert npz_path.endswith('.npz'), "File must be in NPZ format"
        assert split in ['train', 'valid', 'test'], f"Invalid split: {split}"
        
        self.split = split
        self.npz_path = npz_path
        self.map_designs = None
        self.goal_maps = None
        self.opt_policies = None
        self.opt_dists = None
        
        # Load data from NPZ file
        self._load_data()
        
        print(f"\n{'='*60}")
        print(f"Loaded {split.upper()} dataset from: {npz_path}")
        print(f"Number of samples: {len(self)}")
        print(f"Map size: {self.map_designs.shape[1]}x{self.map_designs.shape[2]}")
        print(f"{'='*60}\n")
    
    def _load_data(self):
        """Load arrays from NPZ file based on split."""

        
        with np.load(self.npz_path) as data:
            # NPZ structure: arr_0, arr_1, arr_2, arr_3 for train
            #                arr_4, arr_5, arr_6, arr_7 for valid
            #                arr_8, arr_9, arr_10, arr_11 for test
            split_to_idx = {'train': 0, 'valid': 4, 'test': 8}
            base_idx = split_to_idx[self.split]
            
            # Load the four arrays for this split
            self.map_designs = data[f'arr_{base_idx}'].astype(np.float32)
            self.goal_maps = data[f'arr_{base_idx + 1}'].astype(np.float32)
            self.opt_policies = data[f'arr_{base_idx + 2}'].astype(np.float32)
            self.opt_dists = data[f'arr_{base_idx + 3}'].astype(np.float32)

            print(f"map_designs shape: {self.map_designs.shape}")
            print(f"goal_maps shape: {self.goal_maps.shape}")
            print(f"opt_dists shape: {self.opt_dists.shape}")
        
        # Verify shapes are consistent
        assert self.map_designs.shape[0] == self.goal_maps.shape[0]
        assert self.map_designs.shape[0] == self.opt_dists.shape[0]
        #assert self.map_designs.shape[1:] == self.opt_dists.shape[1:]
    
    def __len__(self):
        """Return number of samples in dataset."""
        return self.map_designs.shape[0]
    
    def __getitem__(self, idx: int):
      """
      Get a single training sample.
      
      Args:
          idx: Index of sample to retrieve
          
      Returns:
          obstacle_map: [1, H, W] - Binary map (1=passable, 0=obstacle)
          goal_map: [1, H, W] - One-hot encoded goal location
          optimal_distances: [1, H, W] - Ground truth distances for supervision
      """
      # map_designs is [H, W], needs channel dimension
      obstacle_map = self.map_designs[idx][np.newaxis, ...]
      
      # goal_maps and opt_dists already have channel dimension [1, H, W]
      goal_map = self.goal_maps[idx]
      optimal_distances = self.opt_dists[idx]
      
      # Convert negative distances to positive (dataset encoding)
      optimal_distances = np.abs(optimal_distances)
      
      return obstacle_map, goal_map, optimal_distances
    
    def get_statistics(self):
      """Compute dataset statistics for analysis."""
      # Use absolute values for distances
      abs_dists = np.abs(self.opt_dists)
      valid_dists = abs_dists[abs_dists > 0]
      
      stats = {
          'num_samples': len(self),
          'map_size': self.map_designs.shape[1:],
          'avg_obstacle_ratio': float(1 - self.map_designs.mean()),
          'avg_optimal_distance': float(valid_dists.mean()) if len(valid_dists) > 0 else 0.0,
          'max_optimal_distance': float(abs_dists.max()),
      }
      return stats


def create_dataloader(npz_path: str, 
                      split: str,
                      batch_size: int,
                      shuffle: bool = True,
                      num_workers: int = 0):
    """
    Create PyTorch DataLoader for path planning dataset.
    
    Args:
        npz_path: Path to NPZ dataset file
        split: 'train', 'valid', or 'test'
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data (typically True for train)
        num_workers: Number of parallel data loading workers
        
    Returns:
        DataLoader yielding batches of (obstacle_maps, goal_maps, optimal_distances)
    """
    dataset = PathPlanningDataset(npz_path, split)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # Speed up GPU transfer
    )


def check_data_loading(npz_path: str, split: str = 'train', num_samples: int = 3):
    """
    Verify data loading works correctly and visualize samples.
    Shows: Obstacles (input), Goal (input), Cost Map (target output)
    
    Args:
        npz_path: Path to NPZ dataset file
        split: Which split to check
        num_samples: Number of samples to display
    """
    import matplotlib.pyplot as plt
    
    print(f"\n{'='*60}")
    print(f"DATA LOADING CHECK")
    print(f"{'='*60}\n")
    
    # Create dataset
    dataset = PathPlanningDataset(npz_path, split)
    
    # Print statistics
    stats = dataset.get_statistics()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Create dataloader
    dataloader = create_dataloader(npz_path, split, batch_size=4, shuffle=False)
    
    # Get one batch
    obstacle_maps, goal_maps, opt_dists = next(iter(dataloader))
    
    print(f"\nBatch Shapes:")
    print(f"  Obstacle maps: {obstacle_maps.shape}")
    print(f"  Goal maps: {goal_maps.shape}")
    print(f"  Optimal distances: {opt_dists.shape}")
    
    print(f"\nValue Ranges:")
    print(f"  Obstacles: [{obstacle_maps.min():.2f}, {obstacle_maps.max():.2f}]")
    print(f"  Goals: [{goal_maps.min():.2f}, {goal_maps.max():.2f}]")
    print(f"  Distances: [{opt_dists.min():.2f}, {opt_dists.max():.2f}]")
    
    # ========== IMPROVED VISUALIZATION ==========
    # Create figure with 3 columns: Obstacles, Goal, Cost Heatmap
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(num_samples, len(obstacle_maps))):
        # Convert to numpy and remove channel dimension
        obstacle_map = obstacle_maps[i, 0].numpy()
        goal_map = goal_maps[i, 0].numpy()
        opt_dist = opt_dists[i, 0].numpy()
        
        # --- Column 1: Obstacle Map (INPUT) ---
        axes[i, 0].imshow(obstacle_map, cmap='gray')
        axes[i, 0].set_title('INPUT 1: Obstacle Map\n(1=passable, 0=obstacle)', fontsize=10)
        axes[i, 0].axis('off')
        
        # --- Column 2: Goal Map (INPUT) ---
        axes[i, 1].imshow(goal_map, cmap='hot')
        axes[i, 1].set_title('INPUT 2: Goal Location\n(one-hot encoded)', fontsize=10)
        axes[i, 1].axis('off')
        
        # --- Column 3: Cost Heatmap (TARGET OUTPUT) - IMPROVED ---
        # Find goal coordinates
        goal_coords = np.where(goal_map == 1)
        if len(goal_coords[0]) > 0:
            goal_y, goal_x = goal_coords[0][0], goal_coords[1][0]
        else:
            goal_y, goal_x = None, None
        
        # Create masked array (obstacles = NaN)
        viz_dist = np.copy(opt_dist)
        obstacle_mask = (obstacle_map == 0)
        viz_dist[obstacle_mask] = np.nan
        
        # Layer 1: Draw obstacles in black
        axes[i, 2].imshow(obstacle_map == 0, cmap='gray_r', vmin=0, vmax=1)
        
        # Layer 2: Overlay cost gradient
        im = axes[i, 2].imshow(viz_dist, cmap='viridis_r', alpha=0.9, interpolation='nearest')
        
        # Mark goal with red 'X'
        if goal_x is not None and goal_y is not None:
            axes[i, 2].scatter(goal_x, goal_y, marker='x', c='red', s=150, 
                              linewidths=3, zorder=10)
        
        axes[i, 2].set_title('OUTPUT: Cost Map (Target)\n(Model should learn this)', fontsize=10)
        axes[i, 2].axis('off')
        
        # Add proper colorbar (doesn't shrink plot)
        divider = make_axes_locatable(axes[i, 2])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax, label="Cost")
    
    plt.tight_layout()
    plt.savefig('data_check_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to data_check_visualization.png")
    print(f"\n{'='*60}")
    print(f"DATA LOADING CHECK COMPLETE ✓")
    print(f"{'='*60}\n")
    
    return dataset, dataloader


if __name__ == "__main__":
    # Example usage
    print("Testing data loader...")
    
    # You'll need to update this path to your actual dataset location
    npz_path = r"E:\Project_CDJ\CDJ\04 - AREAS\UNIVERSITY\PURDUE\SCHOOL\FALL 2025\Artificial Intelligence ECE 57000\CLASS\PROJECTS\astar-efficient-ai\data\bugtrap_forest_064_moore_c16.npz"    
    # Run check
    try:
        dataset, dataloader = check_data_loading(npz_path, split='train', num_samples=3)
        print("\n✓ All checks passed! Data loader is working correctly.")
    except FileNotFoundError:
        print(f"\n✗ File not found: {npz_path}")
        print("Please update the npz_path to your actual dataset location.")
    except Exception as e:
        print(f"\n✗ Error during data loading: {e}")
        raise