
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
# Learned Heuristics for A* Pathfinding 🎯
---
A neural network approach to learning admissible heuristics for A* pathfinding in grid-based environments. This project trains a U-Net with attention gates to predict optimal distances from any cell to a goal, creating obstacle-aware heuristics that dramatically improve search efficiency while maintaining path optimality.
## Motivation
---
Traditional A* pathfinding relies on simple heuristics like Euclidean distance, which work well in open spaces but struggle in environments with complex obstacles. These heuristics are *admissible* (never overestimate true cost), ensuring optimal paths, but they're not very *informative* , they don't account for walls, corridors, or detours.

**The Challenge:** Can we learn heuristics that are both more informative *and* remain admissible?
## Approach
---
We train a neural network to predict the true distance-to-goal for every cell in a grid environment. The key innovation is a custom loss function that heavily penalizes overestimation (which breaks admissibility) while tolerating underestimation (which is safe but less efficient).

**Key Components:**
- **Architecture**: Attention U-Net with skip connections filtered through attention gates
- **Input**: Obstacle map + goal location (2 channels, 64×64)
- **Output**: Predicted distance map (1 channel, 64×64)
- **Loss Function**: Asymmetric loss with 60× penalty for overestimation + boundary-aware weighting to address systematic underestimation near horizontal walls

**Note**: While trained and tested primarily on 64×64 maps, the fully-convolutional architecture can adapt to larger or smaller grids, though further hyperparameter tuning will likely be needed for optimal performance.
## Results
---
🎯 **~87% (Currently) reduction in explored cells** compared to Euclidean heuristic baseline

✅ **0.43% inadmissible cells** (well below the 1% excellence threshold)

✅ **Path optimality maintained** - all paths found are guaranteed optimal

**Small model, big impact** - Achieves deployable results with "only" ~8,000 training samples

The learned heuristic produces A* search patterns that are nearly as efficient as using perfect oracle information (ground truth distances), while requiring only a single forward pass through the network.

![](results/astar_comparison_map1.png)

![](results/astar_comparison_map0.png)
## Code Structure
---

```
📁 src/
├── 📄 dataset.py                      # Dataset loader for NPZ files
├── 📄 learned_heuristic_encoder.py    # U-Net model with attention gates
├── 📄 train_heuristic.py              # Training pipeline with admissible loss
├── 📄 test_model.py                   # Model evaluation and visualization
└── 📄 astar_comparaison.py            # A* comparison framework

📁 visualization/                      # Generated plots and analysis
```

### Model Architecture: `HeuristicCNN`
---
The core model is a U-Net-based architecture with several custom components:

**1. Encoder-Decoder Structure** 🏗️
- **Encoder**: 4 downsampling blocks (64 → 128 → 256 → 512 channels)
- **Bottleneck**: 1024 channels at 16× downsampling
- **Decoder**: 4 upsampling blocks with skip connections

**2. Attention Gates** 🔍 (`AttentionGate`)

Integrated into skip connections to filter encoder features before concatenation:
```python
# Skip connection from encoder is filtered by gating signal from decoder
x_encoder_filtered = attention_gate(decoder_features, encoder_features)
x = concat([x_encoder_filtered, decoder_features])
```

This helps the network focus on relevant spatial features and reduces the influence of noisy or irrelevant encoder activations.

**3. Custom Loss Function** ⚖️ (`AdmissibleDistanceLoss`)

The heart of admissibility enforcement:

```python
loss = overestimation_penalty × overestimation_loss + weighted_underestimation_loss
```

- **Overestimation penalty**: 60× weight (breaks admissibility → heavily penalized)
- **Boundary-aware underestimation**: 5× weight near horizontal walls (addresses systematic bias)
- **Base loss**: L1 distance (more robust than MSE for this task)

The boundary-aware component specifically targets cells adjacent to horizontal walls, where MaxPool-based downsampling tends to cause underestimation.
## 🛠️ Dependencies
---

```bash
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

**Data Format**: NPZ files containing:
- `map_designs`: Binary obstacle maps (1=passable, 0=obstacle)
- `goal_maps`: One-hot encoded goal locations
- `opt_dists`: Ground truth optimal distances (supervision signal)
## 🚀 Usage
---
### 1. Training

Configure hyperparameters in `train_heuristic.py` and run:

```bash
python src/train_heuristic.py
```

**Key hyperparameters:**
```python
OVERESTIMATION_PENALTY = 60      # Higher = stricter admissibility
BOUNDARY_PENALTY = 5.0            # Weight for wall-adjacent cells
LEARNING_RATE = 5e-4              # Conservative for stable training
LOSS_TYPE = 'l1'                  # More robust than MSE
SCHEDULER_PATIENCE = 5            # Epochs before LR reduction
```

### 2. Testing Model Predictions

Visualize learned distance maps on test cases:

```bash
python src/test_model.py
```

Generates a 2×2 grid showing:
- Ground truth distances
- Model predictions
- Absolute error
- Overestimation map (inadmissible regions highlighted in red)

### 3. A* Performance Comparison

Compare learned heuristic against baselines:

```bash
python src/astar_comparaison.py
```

Runs A* with three heuristics:
1. **Combined** (Euclidean + Learned) - our approach
2. **Euclidean** - traditional baseline  
3. **Oracle** (Ground truth) - theoretical upper bound

Output visualization shows:
- Explored cells for each method (colored regions)
- Optimal paths found
- Performance metrics
## 💡 Training Tips
---

**Achieving admissibility** requires careful tuning:
1. **Start conservative**: High overestimation penalty (50-100×) ensures admissibility
2. **Monitor validation**: Track inadmissible ratio, target <1% 
3. **Adjust learning rate**: If inadmissibility increases, reduce LR or increase penalty
4. **Boundary penalties**: If errors concentrate near walls, increase `BOUNDARY_PENALTY`

**Trade-off**: Higher overestimation penalties → better admissibility but potentially weaker heuristic strength. The goal is finding the sweet spot where the model remains admissible while providing meaningful guidance to A*.
## 🔮 Future Directions
---
- Multi-scale supervision for better boundary handling
- Hybrid architectures combining strided convolutions with attention
- Post-processing techniques for guaranteed admissibility
- Extension to dynamic environments and replanning scenarios
- Scaling to larger maps (128×128+) with hierarchical planning

## 📚 Acknowledgments
---
This work builds upon research in learned heuristics for pathfinding:

- **iA∗: Imperative Learning-based A∗ Search for Path Planning**  
  Xiangyu Chen, Fan Yang, and Chen Wang

- **Path Planning using Neural A* Search**  
  Ryo Yonetani, Tatsunori Taniai, Mohammadamin Barekatain, Mai Nishimura, Asako Kanezaki

- **TransPath: Learning Heuristics For Grid-Based Pathfinding via Transformers**  
  Daniil Kirilenko, Anton Andreychuk, Aleksandr Panov, Konstantin Yakovlev

If you want to talk more about this project(And other fun projects), check out my website😉 [danieljacquet.com](https://danieljacquet.com/)