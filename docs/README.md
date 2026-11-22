# Documentation

All public documentation for 2048 DESTROYER is maintained in English. This directory provides engineering and usage guides; benchmark evidence is maintained separately in `benchmarks/`.

## Guides

| Document | Scope |
|---|---|
| [Architecture](architecture.md) | Engine, agents, training/evaluation flow, native backend, GUI boundaries |
| [Training](training.md) | DQN and N-tuple training, checkpoints, validation |
| [Hybrid solver](hybrid.md) | Expectimax, evaluators, budgets, tile downgrading |
| [GPU setup](gpu.md) | ROCm/WSL setup notes and smoke test |
| [Roadmap](roadmap.md) | Planned research and engineering work |

The root [README](../README.md) is the short project overview and quick start. [Benchmark results](../benchmarks/results.md) and [methodology](../benchmarks/methodology.md) document measured performance and the limits of those measurements.
