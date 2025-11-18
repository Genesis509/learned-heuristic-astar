"""
Learned Heuristic Encoder for A* Pathfinding
=============================================
Neural network that predicts distance-to-goal heuristics from obstacle and goal maps.

Input: obstacle_map [B, 1, H, W] + goal_map [B, 1, H, W]
Output: distance_map [B, 1, H, W] - predicted distance from each cell to goal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import PathPlanningDataset


class DoubleConv(nn.Module):
    """Two consecutive conv layers with BatchNorm and ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downsampling block: MaxPool + DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.downsample(x)


class Up(nn.Module):
    """Upsampling block: Upsample + Concatenate + DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: upsampled features from decoder
        x2: skip connection from encoder
        """
        x1 = self.up(x1)

        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class HeuristicCNN(nn.Module):
    """
    UNet-based CNN for learning distance-to-goal heuristics.

    Architecture:
    - Encoder: 4 downsampling blocks (64 → 128 → 256 → 512)
    - Bottleneck: 1024 channels
    - Decoder: 4 upsampling blocks with skip connections
    - Output: 1-channel distance map

    Args:
        in_channels: Number of input channels (default: 2 for obstacle + goal maps)
        base_channels: Base number of channels (default: 64)
    """

    def __init__(self, in_channels=2, base_channels=64):
        super().__init__()

        # Encoder (downsampling path)
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.down4 = Down(base_channels * 8, base_channels * 16)

        # Decoder (upsampling path)
        self.up1 = Up(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.up2 = Up(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.up3 = Up(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up4 = Up(base_channels * 2 + base_channels, base_channels)

        # Output layer
        self.outc = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.ReLU()  # Distances must be non-negative
        )

    def forward(self, obstacle_map, goal_map):
        """
        Forward pass through the network.

        Args:
            obstacle_map: [B, 1, H, W] - Binary obstacle map (1=passable, 0=obstacle)
            goal_map: [B, 1, H, W] - One-hot encoded goal location

        Returns:
            distance_map: [B, 1, H, W] - Preadicted distances to goal
        """
        # Concatenate inputs
        x = torch.cat([obstacle_map, goal_map], dim=1)  # [B, 2, H, W]

        # Encoder
        x1 = self.inc(x)      # [B, 64, H, W]
        x2 = self.down1(x1)   # [B, 128, H/2, W/2]
        x3 = self.down2(x2)   # [B, 256, H/4, W/4]
        x4 = self.down3(x3)   # [B, 512, H/8, W/8]

        # Bottleneck
        x5 = self.down4(x4)   # [B, 1024, H/16, W/16]

        # Decoder with skip connections
        x = self.up1(x5, x4)  # [B, 512, H/8, W/8]
        x = self.up2(x, x3)   # [B, 256, H/4, W/4]
        x = self.up3(x, x2)   # [B, 128, H/2, W/2]
        x = self.up4(x, x1)   # [B, 64, H, W]

        # Output
        distance_map = self.outc(x)  # [B, 1, H, W]

        return distance_map


# ============================================================================
# VISUALIZATION TEST
# ============================================================================

def test_untrained_model():
    """
    Test that the untrained model can generate cost maps.
    Visualizes input (obstacle + goal) and predicted output.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    print("\n" + "="*60)
    print("TESTING UNTRAINED HeuristicCNN")
    print("="*60 + "\n")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeuristicCNN(in_channels=2, base_channels=64).to(device)
    model.eval()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print(f"Device: {device}\n")

    # Create synthetic test data (64x64 map)
    batch_size = 4
    H, W = 64, 64

    # Random obstacle maps
    obstacle_maps = torch.rand(batch_size, 1, H, W) > 0.3  # 30% obstacles
    obstacle_maps = obstacle_maps.float().to(device)

    # Random goal locations
    goal_maps = torch.zeros(batch_size, 1, H, W).to(device)
    for i in range(batch_size):
        goal_y, goal_x = np.random.randint(H), np.random.randint(W)
        goal_maps[i, 0, goal_y, goal_x] = 1.0

    # Run inference
    print("Running forward pass...")
    with torch.no_grad():
        predicted_distances = model(obstacle_maps, goal_maps)

    print(f"Input obstacle_maps shape: {obstacle_maps.shape}")
    print(f"Input goal_maps shape: {goal_maps.shape}")
    print(f"Output distance_maps shape: {predicted_distances.shape}")
    print(f"Output value range: [{predicted_distances.min():.2f}, {predicted_distances.max():.2f}]")

    # ========== VISUALIZATION ==========
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 3*batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)

    for i in range(batch_size):
        obstacle_map = obstacle_maps[i, 0].cpu().numpy()
        goal_map = goal_maps[i, 0].cpu().numpy()
        pred_dist = predicted_distances[i, 0].cpu().numpy()

        # Find goal coordinates
        goal_coords = np.where(goal_map == 1)
        goal_y, goal_x = (goal_coords[0][0], goal_coords[1][0]) if len(goal_coords[0]) > 0 else (None, None)

        # Column 1: Obstacle Map
        axes[i, 0].imshow(obstacle_map, cmap='gray')
        axes[i, 0].set_title('INPUT: Obstacle Map', fontsize=10)
        axes[i, 0].axis('off')

        # Column 2: Goal Map
        axes[i, 1].imshow(goal_map, cmap='hot')
        if goal_x is not None:
            axes[i, 1].scatter(goal_x, goal_y, marker='x', c='cyan', s=100, linewidths=2)
        axes[i, 1].set_title('INPUT: Goal Location', fontsize=10)
        axes[i, 1].axis('off')

        # Column 3: Predicted Distance Map
        # Mask obstacles
        viz_dist = np.copy(pred_dist)
        viz_dist[obstacle_map == 0] = np.nan

        # Plot obstacles as black background
        axes[i, 2].imshow(obstacle_map == 0, cmap='gray_r', vmin=0, vmax=1)

        # Overlay predicted distances
        im = axes[i, 2].imshow(viz_dist, cmap='viridis_r', alpha=0.9, interpolation='nearest')

        # Mark goal
        if goal_x is not None:
            axes[i, 2].scatter(goal_x, goal_y, marker='x', c='red', s=100, linewidths=3, zorder=10)

        axes[i, 2].set_title('OUTPUT: Predicted Distances\n(Untrained Model)', fontsize=10)
        axes[i, 2].axis('off')

        # Add colorbar
        divider = make_axes_locatable(axes[i, 2])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax, label="Distance")

    plt.tight_layout()
    plt.savefig('visualization/untrained_model_test.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: visualization/untrained_model_test.png")

    print("\n" + "="*60)
    print("TEST COMPLETE ✓")
    print("="*60 + "\n")

    return model


if __name__ == "__main__":
    # Run visualization test
    model = test_untrained_model()

    #print("\nModel architecture summary:")
    #print(model)