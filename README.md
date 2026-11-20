# 2048 DESTROYER

An experimental neural-network agent trained to dominate the 4x4 game 2048.

## Current design proposal

- Board: 4x4.
- State: 16 values, each cell encoded as `log2(tile)` (empty = 0), scaled for numerical stability.
- Network: four linear layers total, `16 -> 128 -> 128 -> 64 -> 4`; ReLU between hidden layers.
- Outputs: four Q-values for Up, Down, Left, Right.
- Learning: Deep Q-Network (DQN), with experience replay, target network, epsilon-greedy exploration, and checkpoints.
- Reward: score gained from merges as the primary signal; evaluate small potential-based shaping for empty cells / board organization only if experiments show it helps. Keep raw game score and max tile as the actual evaluation criteria.
- Symmetry: investigate board rotations/reflections for training augmentation and evaluation.

## Project stages

1. **Game engine** — deterministic move/merge logic, spawning separated behind injectable randomness, terminal detection, and tests.
2. **Agent baseline** — random legal agent and simple heuristic baseline to establish reference metrics.
3. **DQN** — four-layer MLP, replay buffer, target updates, exploration schedule, reproducible seeds, checkpoint/resume.
4. **Metrics** — episode score, max tile, duration/moves, merge count, win/2048 rate, distribution plots.
5. **Viewer** — live board display with action/Q-values, training progress charts, and playback of saved episodes.
6. **Experiments** — compare reward variants, input encodings, learning parameters, and optional symmetry augmentation; record findings.

## Reference project

`Q:/PROJECTS/neural-networks-and-deep-learning-modern` is the user's PyTorch educational implementation. Consult its MLP, training, device selection, and test patterns; do not modify it as part of this project.


## Evaluation goal

The fun headline goal is to make the agent *destroy 2048*: reach the 2048 tile consistently, then measure whether it can reliably exceed it. No claim of success until repeated evaluation runs support it.
