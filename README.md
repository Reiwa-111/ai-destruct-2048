# 2048 DESTROYER

A desktop lab for a 4x4 2048 game and a DQN agent. The agent uses a four-linear-layer MLP with 16 inputs and four action-value outputs.


## Run

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH="src"
python -m destroyer.app
```

Or double-click `RUN.bat` (Python and dependencies must already be installed).

## Controls

- Arrow keys: manual moves
- `STEP AI`: one AI move
- `AUTO PLAY`: play continuously
- `TRAIN`: train DQN in the background; stop with `STOP`- `EVALUATE x100`: evaluate a snapshot over 100 games
- Save/load: checkpoint file

## UI

Game board, score/max tile/moves, four live Q-values, training counters, replay-buffer size, loss, rolling 50-episode averages, 2048/4096 rates, score chart, episode log and debug events.


## Network

`16 -> 128 -> 128 -> 64 -> 4`, ReLU on the three hidden layers. Parameter count: **27,204**. The output Q-values correspond to UP, DOWN, LEFT, RIGHT.

DQN includes replay memory, target network, epsilon-greedy exploration, gradient clipping, periodic target sync and checkpointing.


## Notes

This is an early experimental implementation. Evaluation is not proof of learning quality until multiple training runs and baselines are compared. GUI currently uses Tkinter; PyTorch is required for learning.
