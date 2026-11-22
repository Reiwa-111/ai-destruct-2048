#!/usr/bin/env python3
"""Reproducible headless benchmark for the N-tuple + Expectimax hybrid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from destroyer.arena import AGENT_NTUPLE, create_search, evaluate  # noqa: E402
from destroyer.ntuple import NtupleValue  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--node-budget", type=int, default=500_000)
    parser.add_argument("--time-budget-ms", type=float, default=1.0)
    parser.add_argument("--downgrade-threshold", type=int, help="tile value that activates the root-only rank transform")
    parser.add_argument("--downgrade-steps", type=int, default=1, help="fixed to 1 for the research root transform")
    parser.add_argument(
        "--ntuple", type=Path,
        default=ROOT / "models" / "ntuple_v4_correct_50k.npz",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.games < 1 or args.depth < 1 or args.node_budget < 1 or args.time_budget_ms <= 0:
        parser.error("games, depth, node budget and time budget must be positive")
    if args.downgrade_steps != 1:
        parser.error("root tile-downgrading currently supports exactly one rank shift")
    if not args.ntuple.is_file():
        parser.error(f"N-tuple checkpoint does not exist: {args.ntuple}")

    value = NtupleValue.load(args.ntuple)
    search = create_search(
        AGENT_NTUPLE,
        ntuple=value,
        depth=args.depth,
        node_budget=args.node_budget,
        time_budget_ms=args.time_budget_ms,
        downgrade_threshold=args.downgrade_threshold,
        downgrade_steps=args.downgrade_steps,
    )
    assert search is not None
    try:
        print(
            f"backend={search.backend} games={args.games} seed={args.seed} "
            f"depth={args.depth} nodes={args.node_budget} "
            f"time_ms={args.time_budget_ms} downgrade={args.downgrade_threshold}/{args.downgrade_steps} ntuple={args.ntuple}",
            flush=True,
        )
        report = evaluate(
            AGENT_NTUPLE,
            args.games,
            seed_start=args.seed,
            ntuple=value,
            depth=args.depth,
            node_budget=args.node_budget,
            time_budget_ms=args.time_budget_ms,
            downgrade_threshold=args.downgrade_threshold,
            downgrade_steps=args.downgrade_steps,
            search=search,
        )
    finally:
        search.close()

    summary = report.summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if args.output:
        report.save(args.output)
        print(f"report={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
