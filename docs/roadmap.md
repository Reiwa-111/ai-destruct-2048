# Roadmap

This roadmap is research-oriented. Priorities may change when controlled benchmarks show that an approach is ineffective. Items are planned, not implemented features.

## Current baseline

- Deterministic game engine and test coverage for board transitions.
- D4-symmetric N-tuple TD-afterstate evaluator.
- Iterative-deepening Expectimax with native C++ implementation and search budgets.
- N-tuple + Expectimax hybrid with 100-game development results.
- DQN baseline and optional ROCm/WSL training path.

## Next research priorities

1. **Hybrid strength and search quality:** profile search-budget distribution, improve transposition-table efficiency, and investigate adaptive time/node allocation without changing the evaluation contract.
2. **N-tuple training stability:** compare TD(0), TD(λ), temporal-coherence learning rates, and staged training on independent seed sets.
3. **Late-game representation:** investigate rank handling above the current N-tuple table limit and dedicated endgame evaluation.
4. **Root tile-downgrading:** keep disabled until larger paired experiments establish benefit.
5. **Benchmark automation:** standardize checkpoint identifiers/hashes, paired seed comparisons, summary generation, and confidence intervals.
6. **Portability:** improve native build support and exercise Python fallback parity on more environments.

## Validation gates

- No feature is promoted to the default hybrid without paired comparisons on a fixed, previously unseen seed set.
- Report confidence intervals for tile-attainment rates and paired score differences.
- Record search configuration, model identifier, checkpoint hash, environment, elapsed time, and per-decision latency distributions.
- Run regression tests before and after native or game-engine changes.
- Do not claim the 99% 2048 target until the independent evaluation supports that threshold.
