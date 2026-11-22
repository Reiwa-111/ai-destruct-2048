# Architecture

## Overview

DESTROYER separates the 2048 rules, agent decision logic, training/evaluation, the native C++ acceleration layer, and the Tkinter user interface. Headless benchmark runners use the same engine and agent code as the GUI, avoiding a second game implementation for experiments.

```text
                  +------------------------------+
                  |          Tkinter GUI         |
                  | play / train / evaluate     |
                  +---------------+--------------+
                                  |
        +------------------------+-------------------------+
        |             Python application layer           |
        |       agents / arena / records / search       |
        +---------------+------------------+--------------+
                        |                  |
                 +------+-------+  +--------+-----------+
                 | Game engine |  | Native C++ kernels |
                 | rules/spawn|  | N-tuple / search  |
                 +------------+  +-------------------+
```

## Game engine

`src/destroyer/game.py` implements the 4x4 board and standard 2048 mechanics. A move is split into deterministic slide/merge (`apply_move`) and random tile spawn (`spawn_tile`). The engine also provides exact spawn outcomes for search.

The spawn distribution is 90% tile 2 and 10% tile 4, uniformly selected among empty cells. A seeded `random.Random` instance controls spawns; the undo history saves RNG state so undo/replay can reproduce the same tile. Board inputs are validated and frozen to tuples for cached move operations.

## Agents and value functions

- `agent.py`: DQN model, replay buffer, legal-action masking, checkpoint loading, and training updates.
- `ntuple.py`: N-tuple value tables, D4 symmetry feature indexing, afterstate learning, episode training, and position-pool management.
- `search.py`: Expectimax, positional heuristic, search limits, and native-library adapter.
- `arena.py`: common agent construction, episode execution, and evaluation report generation.

## Native backend

The optional `native/libdestroyer_ntuple.so` is built from `native/ntuple.cpp` and `native/search.cpp`. Python loads it through `ctypes`. It accelerates tuple evaluation/updates, episode training, move evaluation, and hybrid search. If the library is missing or cannot be loaded, Python fallback paths remain available.

The native build script targets Linux and uses `-O3 -march=native`; the resulting binary is host-specific and is not committed.

## Training and evaluation separation

Training episodes can update N-tuple weights and collect position samples. Periodic greedy validation uses fresh games with no exploration, restart sampling, or online updates. Benchmark scripts are headless, seeded, and write per-game JSON records or summaries. Search counters and decision-latency statistics are collected separately from game score.

## GUI

`src/destroyer/ui/` contains the Tkinter frontend. GUI tasks run outside the UI event loop where applicable, and evaluation can be cancelled. Game records and UI state are stored locally in `models/`; they are runtime state, not source or benchmark evidence.

## Repository data policy

- Track source code, tests, scripts, and curated Markdown benchmark reports.
- Do not commit model checkpoints, replay buffers, local records, UI preferences, raw exploratory logs, generated shared libraries, or environment-specific build output.
- Published benchmarks should state seed ranges, checkpoint identity when available, configuration, sample size, and uncertainty. A benchmark without its model checkpoint is historical evidence, not a fully reproducible experiment.
