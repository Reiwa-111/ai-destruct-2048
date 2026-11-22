#!/usr/bin/env python3
"""Resume DQN training on the WSL ROCm GPU, then play one greedy game."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("HSA_ENABLE_DXG_DETECTION", "1")
os.environ.setdefault("REIWA_DEVICE", "cuda")
os.environ.setdefault("REIWA_REQUIRE_ROCM", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from destroyer import ACTIONS, DQNAgent, Game2048, encode  # noqa: E402
from destroyer.arena import AGENT_DQN, play_game  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, default=ROOT / "models" / "latest.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "dqn_rocm_finetuned.pt")
    parser.add_argument("--steps", type=int, default=50_000, help="New environment interactions")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--updates-per-round", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20_481)
    parser.add_argument("--eval-seed", type=int, default=2_048_000)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    args = parser.parse_args()

    if min(args.steps, args.envs, args.updates_per_round, args.checkpoint_every) < 1:
        parser.error("steps, envs, updates-per-round and checkpoint-every must be positive")
    if not args.resume.is_file():
        parser.error(f"resume checkpoint does not exist: {args.resume}")

    if torch.version.hip is None or not torch.version.hip.startswith("6.4."):
        raise SystemExit(f"Expected ROCm/HIP 6.4.x PyTorch, got HIP={torch.version.hip!r}")
    probe_started = time.perf_counter()
    print("[DQN] Probing WSL GPU (first HIP initialization may take ~20s)...", flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("ROCm GPU is unavailable; CPU training fallback is disabled.")
    torch.cuda.init()
    device = torch.device("cuda")
    print(
        f"[DQN] GPU={torch.cuda.get_device_name(device)} HIP={torch.version.hip} "
        f"torch={torch.__version__} probe={time.perf_counter() - probe_started:.2f}s",
        flush=True,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    agent = DQNAgent(device=device)
    prior_episodes = int(agent.load(args.resume))
    if agent.device != device or not agent.online.layers[0].weight.is_cuda:
        raise SystemExit("Loaded DQN model is not resident on the ROCm GPU.")
    if not agent.replay.states.is_cuda:
        raise SystemExit("Loaded replay buffer is not resident on the ROCm GPU.")
    print(
        f"[DQN] Resumed {args.resume.name}: episodes={prior_episodes} "
        f"env_steps={agent.environment_steps} updates={agent.steps} "
        f"epsilon={agent.epsilon:.4f} replay={len(agent.replay)} "
        f"lr={agent.learning_rate:g}",
        flush=True,
    )

    envs = [Game2048(random.Random(args.seed + index)) for index in range(args.envs)]
    episode_scores = [0] * args.envs
    episodes_completed = 0
    collected_scores: list[int] = []
    collected_tiles: list[int] = []
    losses: list[float] = []
    started = time.perf_counter()
    last_saved_at = 0
    new_steps = 0

    while new_steps < args.steps:
        for index, game in enumerate(envs):
            if new_steps >= args.steps:
                break
            legal = game.legal_actions()
            if not legal:
                # A terminal game is replaced immediately; this is defensive.
                envs[index] = Game2048(random.Random(args.seed + 1009 * (episodes_completed + index + 1)))
                episode_scores[index] = 0
                game = envs[index]
                legal = game.legal_actions()

            state = encode(game.board)
            action_index = agent.act(game.board, legal, explore=True)
            action = ACTIONS[action_index]
            step = game.move(action)
            done = step.game_over
            next_legal = [] if done else game.legal_actions()
            next_mask = [candidate in next_legal for candidate in ACTIONS]
            agent.replay.add(
                state,
                action_index,
                float(step.score_delta),
                encode(game.board),
                done,
                next_mask,
            )
            new_steps += 1
            agent.environment_steps += 1
            agent.update_epsilon()
            episode_scores[index] += step.score_delta

            if done:
                episodes_completed += 1
                collected_scores.append(episode_scores[index])
                collected_tiles.append(game.max_tile)
                envs[index] = Game2048(
                    random.Random(args.seed + 1009 * (episodes_completed + args.envs))
                )
                episode_scores[index] = 0

        if len(agent.replay) >= max(agent.batch, agent.warmup):
            for _ in range(args.updates_per_round):
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)

        if new_steps % 1_000 < args.envs or new_steps == args.steps:
            elapsed = time.perf_counter() - started
            recent_scores = collected_scores[-100:]
            recent_tiles = collected_tiles[-100:]
            recent_loss = losses[-500:]
            print(
                f"[DQN] env={new_steps}/{args.steps} episodes={episodes_completed} "
                f"updates={agent.steps} epsilon={agent.epsilon:.4f} "
                f"loss={float(np.mean(recent_loss)) if recent_loss else float('nan'):.4f} "
                f"score100={float(np.mean(recent_scores)) if recent_scores else float('nan'):.1f} "
                f"tile100={float(np.mean(recent_tiles)) if recent_tiles else float('nan'):.1f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

        if new_steps - last_saved_at >= args.checkpoint_every:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            agent.save(args.output, prior_episodes + episodes_completed, include_replay=True)
            last_saved_at = new_steps
            print(f"[DQN] checkpoint saved: {args.output}", flush=True)

    torch.cuda.synchronize(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.output, prior_episodes + episodes_completed, include_replay=True)
    training_elapsed = time.perf_counter() - started
    print(
        f"[DQN] TRAINING_DONE env_steps={new_steps} episodes={episodes_completed} "
        f"updates={agent.steps} epsilon={agent.epsilon:.4f} "
        f"replay={len(agent.replay)} elapsed={training_elapsed:.1f}s",
        flush=True,
    )

    # Exactly one held-out, greedy game. No epsilon exploration or online learning.
    game_started = time.perf_counter()
    result = play_game(AGENT_DQN, args.eval_seed, dqn_agent=agent)
    eval_elapsed = time.perf_counter() - game_started
    report = {
        "agent": AGENT_DQN,
        "checkpoint": str(args.output),
        "resume_checkpoint": str(args.resume),
        "training": {
            "new_environment_steps": new_steps,
            "episodes_completed": episodes_completed,
            "optimizer_updates_total": agent.steps,
            "epsilon": agent.epsilon,
            "replay_size": len(agent.replay),
            "training_seconds": training_elapsed,
            "mean_training_episode_score": float(np.mean(collected_scores)) if collected_scores else None,
            "mean_training_episode_max_tile": float(np.mean(collected_tiles)) if collected_tiles else None,
            "mean_recent_loss": float(np.mean(losses[-500:])) if losses else None,
        },
        "evaluation": {**asdict(result), "elapsed_s": eval_elapsed},
    }
    report_path = ROOT / "reports" / f"dqn_rocm_one_game_seed{args.eval_seed}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[DQN] ONE_GAME seed={result.seed} score={result.score:,} "
        f"max_tile={result.max_tile:,} moves={result.moves:,} "
        f"time={eval_elapsed:.2f}s report={report_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
