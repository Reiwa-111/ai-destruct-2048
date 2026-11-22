# 2048 DESTROYER

**2048 DESTROYER** is an experimental AI lab for the 2048 puzzle. It combines learned value functions, search, a native C++ backend, reproducible evaluation tools, and a desktop GUI. The primary research direction is a stronger N-tuple + Expectimax hybrid; DQN is retained as a separate baseline.

> Experimental research software. Results are development measurements, not a claim of state-of-the-art play or a statistically established 99% success rate.

## Features

- Deterministic 4×4 game engine with seeded tile spawning, exact afterstates, undo, and game records.
- DQN baseline with Double DQN targets, masked legal actions, replay, and device-aware training.
- D4-symmetric N-tuple value function with TD-afterstate training and optional TD(λ) traces.
- Iterative-deepening Expectimax with exact spawn probabilities, transposition caching, node/time budgets, and learned or heuristic evaluation.
- N-tuple + Expectimax hybrid, including an experimental root-only tile-downgrading option.
- Native C++ kernels for N-tuple operations and search; Python implementations remain available as fallbacks.
- Tkinter desktop interface for manual play, agents, training, evaluation, and saved records.
- Seeded headless benchmarks with JSON reports.

## Quick start

### Windows

Requirements: Python 3.10+ and pip. Install dependencies:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

Double-click `RUN.bat` or run it from PowerShell:

```powershell
./RUN.bat
```

Run tests:

```powershell
python -m pytest -q
```

The native Linux C++ shared library is not used on Windows; search can use the Python fallback.

### Linux / WSL

On Ubuntu, install Tkinter if it is not already available (
`sudo apt install python3-tk`). Install dependencies and the editable package:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e '.[dev]'
python3 -m pytest -q
PYTHONPATH=src python3 -m destroyer.app
```

For the native search backend, build the optional shared library:

```bash
bash scripts/build_native.sh
```

The build requires GCC/G++ and produces `native/libdestroyer_ntuple.so`, which is ignored by Git. Native search is currently Linux-oriented.

### Optional AMD GPU training in WSL

The DQN training path can use PyTorch ROCm through WSL. The provided launcher targets the locally tested ROCm 6.4.x environment and intentionally fails rather than silently falling back to CPU:

```bat
RUN_GPU_WSL.bat
```

Read [GPU setup and validation](docs/gpu.md) before attempting this environment. GPU support applies to DQN tensor operations; N-tuple training and Expectimax search use CPU/native code.

## Agents

| Agent | Description | Status |
|---|---|---|
| Random | Uniform legal move selection | Baseline |
| Greedy N-tuple | Selects the move with the best learned afterstate value | Experimental learned evaluator |
| Classic Expectimax | Search with a hand-crafted positional heuristic | Search baseline |
| N-tuple + Expectimax | Search evaluated by the trained N-tuple value function | Main hybrid |
| DQN | Small neural Q-network | Separate baseline; not the current strongest agent |

## Latest benchmark snapshot

The current recorded hybrid holdout uses a continued N-tuple checkpoint trained for 1M additional episodes, depth cap 3, 500,000-node cap, and 1 ms search budget over 100 fresh games (seeds `1000000–1000099`).

| Metric | Result |
|---|---:|
| Mean score | 190,098.6 |
| Median score | 175,480 |
| Reached 2048 | 100/100 |
| Reached 4096 | 100/100 |
| Reached 8192 | 89/100 |
| Reached 16384 | 28/100 |
| Mean decision latency | 1.027 ms |
| Decision-latency P95 | 1.085 ms |

The observed 100% 2048 rate has a Wilson 95% confidence interval of 96.3–100.0%; this sample does not establish a 99% lower confidence bound. One recorded decision latency reached 107 ms, so mean/P95 latency should not be interpreted as a hard real-time guarantee. Check [benchmark results](benchmarks/results.md) and [methodology](benchmarks/methodology.md) for context and other experiments.

## Documentation

- [Documentation index](docs/README.md)
- [Architecture](docs/architecture.md) — engine, agents, native boundary, GUI.
- [Training](docs/training.md) — DQN and N-tuple workflows.
- [Hybrid solver](docs/hybrid.md) — search, budgets, evaluators, and experimental options.
- [GPU/ROCm setup](docs/gpu.md) — WSL environment, smoke test, limitations.
- [Benchmark methodology](benchmarks/methodology.md) — reproducible evaluation contract.
- [Benchmark results](benchmarks/results.md) — current results and caveats.
- [Roadmap](docs/roadmap.md) — research directions and engineering priorities.

## Repository layout

```text
src/destroyer/   Game engine, agents, search, training, and GUI
native/          C++ N-tuple/search kernels (source only)
scripts/         Training, benchmark, build, and diagnostic commands
tests/           Game, agent, GUI-record, and hybrid regression tests
docs/            English project and engineering documentation
benchmarks/       Curated benchmark methodology and result summaries
models/           Local checkpoints and runtime state (not versioned)
reports/          Raw experiment outputs (not versioned)
```

## License

Apache-2.0. See [LICENSE](LICENSE).
