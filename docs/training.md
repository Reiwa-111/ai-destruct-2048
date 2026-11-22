# Training

DESTROYER contains two learning baselines: DQN and a tabular N-tuple value function. The N-tuple model is the primary evaluator used by the hybrid solver.

## N-tuple training

The evaluator uses four six-cell patterns, each shared across the eight rotations/reflections of the board (D4 symmetry). The tables are float32 and contain 16 cell states (empty and tile ranks through 32768). Each pattern table contains `16^6` weights; the four tables occupy about 256 MiB.

The trainer uses afterstate TD updates. `--td-lambda` enables the experimental eligibility-trace implementation; `0.0` preserves TD(0). Fresh-start training is the default. Position capture and restart sampling are optional and intended for later-stage continuation experiments.

Fresh training:

```bash
python scripts/train_ntuple.py --episodes 700000 --seed 2048 --output models/ntuple_latest.npz
```

Continue a checkpoint:

```bash
python scripts/train_ntuple.py \
  --resume models/ntuple_latest.npz \
  --episodes 100000 \
  --output models/ntuple_continued.npz
```

Train with native acceleration when available:

```bash
bash scripts/build_native.sh
python scripts/train_ntuple.py --episodes 700000 --output models/ntuple_latest.npz
```

The output checkpoint and position pool are local files. The project does not currently publish a canonical downloadable checkpoint.

## Greedy N-tuple evaluation

Use the separate evaluator for a fixed seed set. It does not update weights or use exploratory actions:

```bash
python scripts/benchmark_ntuple.py \
  --checkpoint models/ntuple_latest.npz \
  --games 100 \
  --seed 5000000 \
  --output reports/ntuple_holdout.json
```

Evaluation games are fresh-start games. Report tile attainment rates, scores, moves, and confidence intervals; training-window averages are not a substitute for holdout results.

## DQN

The compact DQN uses a 16-input MLP with four 128-unit hidden layers and four action outputs. The implementation uses Double DQN targets, legal-next-action masks, a replay buffer, Huber loss, gradient clipping, and soft target updates. A larger legacy network remains loadable for old checkpoints.

The WSL GPU trainer resumes a checkpoint, trains on the selected PyTorch ROCm device, saves a new checkpoint, and then runs one greedy evaluation game. That single post-training game is only a smoke check, not a strength evaluation:

```bash
python scripts/train_dqn.py --steps 50000
```

## Checkpoints and experiment hygiene

- Keep the source checkpoint intact; write continuations to a new output path.
- Record seed, episode/environment-step count, learning parameters, checkpoint hash, and validation seed range in experiment notes.
- Use a holdout seed range that is not reused for checkpoint selection.
- Do not promote a checkpoint based only on its training score or loss.
- Model files are ignored by Git because they are large research artifacts and may contain replay state.
