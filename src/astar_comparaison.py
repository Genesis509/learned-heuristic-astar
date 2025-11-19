"""
A* Pathfinding Comparison: Learned Heuristic vs Euclidean Distance
Evaluates trained heuristic network against baseline Euclidean heuristic.
"""

import heapq
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dataset import PathPlanningDataset
from learned_heuristic_encoder import HeuristicCNN


class AStarPathfinder:
    """A* pathfinding with pluggable heuristic functions."""
    
    def __init__(self, obstacle_map, goal_pos, heuristic_fn):
        self.obstacle_map = obstacle_map
        self.goal = goal_pos
        self.heuristic_fn = heuristic_fn
        self.h, self.w = obstacle_map.shape
        
    def get_neighbors(self, pos):
        """8-connected Moore neighborhood with diagonal movement."""
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = pos[0] + dy, pos[1] + dx
                if 0 <= ny < self.h and 0 <= nx < self.w and self.obstacle_map[ny, nx] > 0:
                    cost = 1.414213 if abs(dy) + abs(dx) == 2 else 1.0
                    neighbors.append(((ny, nx), cost))
        return neighbors
    
    def search(self, start_pos):
        """Execute A* search from start to goal."""
        counter = 0
        open_set = [(0, counter, start_pos)]
        came_from = {}
        g_score = {start_pos: 0}
        explored = set()
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            if current in explored:
                continue
            explored.add(current)
            
            if current == self.goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_pos)
                return path[::-1], g_score[self.goal], explored
            
            for neighbor, edge_cost in self.get_neighbors(current):
                tentative_g = g_score[current] + edge_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic_fn(neighbor)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))
        
        return None, None, explored


def euclidean_heuristic_factory(goal_pos):
    """Euclidean distance heuristic."""
    def heuristic(pos):
        dy = pos[0] - goal_pos[0]
        dx = pos[1] - goal_pos[1]
        return np.sqrt(dy*dy + dx*dx)
    return heuristic


def learned_heuristic_factory(distance_map):
    """Lookup-based learned heuristic from neural network predictions."""
    def heuristic(pos):
        return float(distance_map[pos[0], pos[1]])
    return heuristic


def combined_heuristic_factory(euclidean_fn, learned_map):
    """Combined heuristic: f(n) = g(n) + h_euclidean(n) + h_learned(n)"""
    def heuristic(pos):
        return euclidean_fn(pos) + float(learned_map[pos[0], pos[1]])
    return heuristic


def find_valid_start_goal(obstacle_map, min_distance=20):
    """Find start and goal positions with sufficient separation."""
    passable = np.argwhere(obstacle_map > 0)
    
    if len(passable) < 2:
        raise ValueError("Not enough passable cells")
    
    for _ in range(100):
        idx = np.random.choice(len(passable), 2, replace=False)
        start = tuple(passable[idx[0]])
        goal = tuple(passable[idx[1]])
        
        dist = np.sqrt((start[0]-goal[0])**2 + (start[1]-goal[1])**2)
        if dist >= min_distance:
            return start, goal
    
    return tuple(passable[0]), tuple(passable[-1])


def visualize_comparison(obstacle_map, goal, 
                        path_learned, explored_learned, cost_learned,
                        path_euclidean, explored_euclidean, cost_euclidean,
                        path_oracle, explored_oracle, cost_oracle,
                        pred_distances, target_distances,
                        save_path='astar_comparison.png'):
    """Create side-by-side visualization of all three A* variants."""
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    obstacle_mask = (obstacle_map == 0)
    
    # Common: Ground truth distances
    target_viz = np.copy(target_distances)
    target_viz[obstacle_mask] = np.nan
    
    pred_viz = np.copy(pred_distances)
    pred_viz[obstacle_mask] = np.nan
    
    # ========== LEFT COLUMN: LEARNED HEURISTIC ==========
    
    # Top-left: Learned heuristic exploration
    ax_learned = axes[0, 0]
    ax_learned.imshow(obstacle_mask, cmap='gray_r', vmin=0, vmax=1)
    
    explored_map_learned = np.zeros_like(obstacle_map)
    for pos in explored_learned:
        explored_map_learned[pos[0], pos[1]] = 1
    explored_map_learned[obstacle_mask] = np.nan
    
    im1 = ax_learned.imshow(explored_map_learned, cmap='Blues', alpha=0.6, vmin=0, vmax=1)
    
    if path_learned:
        path_y = [p[0] for p in path_learned]
        path_x = [p[1] for p in path_learned]
        ax_learned.plot(path_x, path_y, 'r-', linewidth=3, label='Path', zorder=10)
        ax_learned.scatter(path_x[0], path_y[0], c='green', s=200, marker='o', 
                          edgecolors='black', linewidths=2, label='Start', zorder=11)
    
    ax_learned.scatter(goal[1], goal[0], c='red', s=200, marker='*', 
                      edgecolors='black', linewidths=2, label='Goal', zorder=11)
    
    title_learned = f'Combined Heuristic A* (Euclidean + Learned)\nCost: {cost_learned:.2f} | Explored: {len(explored_learned)} cells'
    ax_learned.set_title(title_learned, fontsize=13, fontweight='bold', color='blue')
    ax_learned.axis('off')
    ax_learned.legend(loc='upper right', fontsize=9)
    
    # Bottom-left: Predicted distance map
    ax_pred = axes[1, 0]
    ax_pred.imshow(obstacle_mask, cmap='gray_r', vmin=0, vmax=1)
    im2 = ax_pred.imshow(pred_viz, cmap='viridis_r', alpha=0.9, interpolation='nearest')
    ax_pred.scatter(goal[1], goal[0], c='red', s=200, marker='*', 
                   edgecolors='black', linewidths=2, zorder=11)
    ax_pred.set_title('Learned Distance Map (Neural Network)', fontsize=13, fontweight='bold')
    ax_pred.axis('off')
    
    divider2 = make_axes_locatable(ax_pred)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im2, cax=cax2, label="Distance")
    
    # ========== MIDDLE COLUMN: EUCLIDEAN HEURISTIC ==========
    
    # Top-middle: Euclidean exploration
    ax_euclidean = axes[0, 1]
    ax_euclidean.imshow(obstacle_mask, cmap='gray_r', vmin=0, vmax=1)
    
    explored_map_euclidean = np.zeros_like(obstacle_map)
    for pos in explored_euclidean:
        explored_map_euclidean[pos[0], pos[1]] = 1
    explored_map_euclidean[obstacle_mask] = np.nan
    
    im3 = ax_euclidean.imshow(explored_map_euclidean, cmap='Oranges', alpha=0.6, vmin=0, vmax=1)
    
    if path_euclidean:
        path_y = [p[0] for p in path_euclidean]
        path_x = [p[1] for p in path_euclidean]
        ax_euclidean.plot(path_x, path_y, 'r-', linewidth=3, label='Path', zorder=10)
        ax_euclidean.scatter(path_x[0], path_y[0], c='green', s=200, marker='o',
                            edgecolors='black', linewidths=2, label='Start', zorder=11)
    
    ax_euclidean.scatter(goal[1], goal[0], c='red', s=200, marker='*',
                        edgecolors='black', linewidths=2, label='Goal', zorder=11)
    
    title_euclidean = f'Euclidean Heuristic A*\nCost: {cost_euclidean:.2f} | Explored: {len(explored_euclidean)} cells'
    ax_euclidean.set_title(title_euclidean, fontsize=13, fontweight='bold', color='darkorange')
    ax_euclidean.axis('off')
    ax_euclidean.legend(loc='upper right', fontsize=9)
    
    # Bottom-middle: Euclidean distance visualization (empty/informative)
    ax_euclidean_viz = axes[1, 1]
    ax_euclidean_viz.text(0.5, 0.5, 'Euclidean Distance\n(Straight-Line)\n\nBaseline Heuristic', 
                          ha='center', va='center', fontsize=14, 
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_euclidean_viz.axis('off')
    
    # ========== RIGHT COLUMN: GROUND TRUTH ORACLE ==========
    
    # Top-right: Oracle exploration
    ax_oracle = axes[0, 2]
    ax_oracle.imshow(obstacle_mask, cmap='gray_r', vmin=0, vmax=1)
    
    explored_map_oracle = np.zeros_like(obstacle_map)
    for pos in explored_oracle:
        explored_map_oracle[pos[0], pos[1]] = 1
    explored_map_oracle[obstacle_mask] = np.nan
    
    im4 = ax_oracle.imshow(explored_map_oracle, cmap='Greens', alpha=0.6, vmin=0, vmax=1)
    
    if path_oracle:
        path_y = [p[0] for p in path_oracle]
        path_x = [p[1] for p in path_oracle]
        ax_oracle.plot(path_x, path_y, 'r-', linewidth=3, label='Path', zorder=10)
        ax_oracle.scatter(path_x[0], path_y[0], c='green', s=200, marker='o',
                         edgecolors='black', linewidths=2, label='Start', zorder=11)
    
    ax_oracle.scatter(goal[1], goal[0], c='red', s=200, marker='*',
                     edgecolors='black', linewidths=2, label='Goal', zorder=11)
    
    title_oracle = f'Oracle (Ground Truth) A*\nCost: {cost_oracle:.2f} | Explored: {len(explored_oracle)} cells'
    ax_oracle.set_title(title_oracle, fontsize=13, fontweight='bold', color='green')
    ax_oracle.axis('off')
    ax_oracle.legend(loc='upper right', fontsize=9)
    
    # Bottom-right: Ground truth distance map
    ax_target = axes[1, 2]
    ax_target.imshow(obstacle_mask, cmap='gray_r', vmin=0, vmax=1)
    im5 = ax_target.imshow(target_viz, cmap='viridis_r', alpha=0.9, interpolation='nearest')
    ax_target.scatter(goal[1], goal[0], c='red', s=200, marker='*',
                     edgecolors='black', linewidths=2, zorder=11)
    ax_target.set_title('Ground Truth Distance Map', fontsize=13, fontweight='bold')
    ax_target.axis('off')
    
    divider5 = make_axes_locatable(ax_target)
    cax5 = divider5.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im5, cax=cax5, label="Distance")
    
    # Performance summary
    if path_learned and path_euclidean and path_oracle:
        learned_vs_euclidean = 100 * (len(explored_euclidean) - len(explored_learned)) / len(explored_euclidean)
        learned_vs_oracle = 100 * (len(explored_learned) - len(explored_oracle)) / len(explored_oracle)
        summary = (f'Performance: Combined reduced exploration by {learned_vs_euclidean:.1f}% vs Euclidean | '
                  f'Oracle is {abs(learned_vs_oracle):.1f}% {"better" if learned_vs_oracle > 0 else "worse"} than Combined')
        fig.suptitle(summary, fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compare_astar_methods(model_path, data_path, map_index=0, split='test', 
                         save_prefix='comparison'):
    """
    Compare learned heuristic A* against Euclidean baseline.
    
    Args:
        model_path: Path to trained model checkpoint
        data_path: Path to NPZ dataset
        map_index: Index of map to test
        split: Dataset split to use
        save_prefix: Prefix for saved visualizations
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset = PathPlanningDataset(data_path, split=split)
    if map_index >= len(dataset):
        print(f"ERROR: Index {map_index} out of range (max: {len(dataset)-1})")
        return
    
    obstacle_map_t, goal_map_t, target_distances_t = dataset[map_index]
    
    # Load trained model
    model = HeuristicCNN(in_channels=2, base_channels=64).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except FileNotFoundError:
        print(f"ERROR: Model not found at {model_path}")
        return
    
    # Generate learned distance predictions
    with torch.no_grad():
        obstacle_map_batch = torch.from_numpy(obstacle_map_t).unsqueeze(0).to(device)
        goal_map_batch = torch.from_numpy(goal_map_t).unsqueeze(0).to(device)
        pred_distances_batch = model(obstacle_map_batch, goal_map_batch)
    
    # Convert to numpy
    obstacle_map = obstacle_map_t[0]
    goal_map = goal_map_t[0]
    target_distances = target_distances_t[0]
    pred_distances = pred_distances_batch.squeeze().cpu().numpy()
    
    # Find goal position
    goal_coords = np.argwhere(goal_map == 1)
    if len(goal_coords) == 0:
        print("ERROR: No goal found in goal map")
        return
    goal_pos = tuple(goal_coords[0])
    
    # Find valid start position
    start_pos, _ = find_valid_start_goal(obstacle_map, min_distance=15)
    
    # Create euclidean heuristic
    euclidean_heuristic = euclidean_heuristic_factory(goal_pos)
    
    # ========== Run A* with COMBINED heuristic (Euclidean + Learned) ==========
    combined_heuristic = combined_heuristic_factory(euclidean_heuristic, pred_distances)
    astar_learned = AStarPathfinder(obstacle_map, goal_pos, combined_heuristic)
    path_learned, cost_learned, explored_learned = astar_learned.search(start_pos)
    
    if path_learned is None:
        print(f"WARNING: Combined heuristic found no path (explored {len(explored_learned)} cells)")
    
    # ========== Run A* with Euclidean heuristic only ==========
    astar_euclidean = AStarPathfinder(obstacle_map, goal_pos, euclidean_heuristic)
    path_euclidean, cost_euclidean, explored_euclidean = astar_euclidean.search(start_pos)
    
    if path_euclidean is None:
        print(f"WARNING: Euclidean heuristic found no path (explored {len(explored_euclidean)} cells)")
    
    # ========== Run A* with ground truth oracle ==========
    oracle_heuristic = combined_heuristic_factory(euclidean_heuristic, target_distances)
    astar_oracle = AStarPathfinder(obstacle_map, goal_pos, oracle_heuristic)
    path_oracle, cost_oracle, explored_oracle = astar_oracle.search(start_pos)
    
    if path_oracle is None:
        print(f"WARNING: Oracle heuristic found no path (explored {len(explored_oracle)} cells)")
    
    # Visualize results
    save_path = f'{save_prefix}_map{map_index}.png'
    visualize_comparison(
        obstacle_map, goal_pos,
        path_learned, explored_learned, cost_learned,
        path_euclidean, explored_euclidean, cost_euclidean,
        path_oracle, explored_oracle, cost_oracle,
        pred_distances, target_distances,
        save_path
    )
    
    print(f"✓ Saved: {save_path}")
    
    # Print performance metrics only if all succeeded
    if path_learned and path_euclidean and path_oracle:
        learned_vs_euclidean = 100 * (len(explored_euclidean) - len(explored_learned)) / len(explored_euclidean)
        learned_vs_oracle = 100 * (len(explored_learned) - len(explored_oracle)) / len(explored_oracle)
        print(f"  Combined: {len(explored_learned)} | Euclidean: {len(explored_euclidean)} | Oracle: {len(explored_oracle)} cells")
        print(f"  Combined vs Euclidean: {learned_vs_euclidean:.1f}% reduction")
        print(f"  Combined vs Oracle: {abs(learned_vs_oracle):.1f}% {'worse' if learned_vs_oracle > 0 else 'better'}")



if __name__ == "__main__":
    MODEL_PATH = 'models/good_model_adaptive_scheduler.pth'
    DATA_PATH = 'data/mpd/instances/064/bugtrap_forest_064_moore_c16.npz'
    
    # Compare on two test maps
    compare_astar_methods(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        map_index=0,
        split='test',
        save_prefix='astar_comparison'
    )
    
    compare_astar_methods(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        map_index=1,
        split='test',
        save_prefix='astar_comparison'
    )