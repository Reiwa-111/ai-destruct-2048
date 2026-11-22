# Benchmark results

These are selected development measurements, not state-of-the-art claims. Trained checkpoints and raw reports remain local and are not distributed in this repository. The latest run records checkpoint and native-library hashes plus the source revision so its provenance is explicit; older experiments have weaker provenance and should be treated as historical measurements.

## Latest hybrid holdout: current N-tuple checkpoint, 100 games

This is the current primary result. The agent uses the latest locally available N-tuple checkpoint, a native C++ evaluator/search backend, and Expectimax with the standard spawn distribution. Evaluation starts from fresh boards, performs no learning or exploration, and uses a new 100-seed range.

| Metric | Result |
|---|---:|
| Checkpoint | `ntuple_fresh_continue1m5_more500k.npz` |
| Games | 100 (complete; not cancelled) |
| Seed range | 9,100,000–9,100,099 |
| Search depth cap | 3 player decisions |
| Node budget | 500,000 |
| Time budget target | 1.0 ms/decision |
| Root tile downgrading | Disabled |
| Mean score | **208,339.9** |
| Median score | 177,018 |
| Score standard deviation | 61,005.7 |
| Score range | 80,356–368,108 |
| Mean moves/game | 8,327.16 |
| Reached 2048 | 100/100 (100%) |
| Reached 4096 | 100/100 (100%) |
| Reached 8192 | 99/100 (99%) |
| Reached 16384 | 36/100 (36%) |
| Maximum tile | 16384 |
| Mean search nodes/game | 13,851,767 |
| Mean transposition-cache hits/game | 2,209,700 |
| Mean completed search depth | 2.059 |
| Mean decision latency | 1.041 ms |
| Mean per-game decision-latency P95 | 1.091 ms |
| Maximum recorded decision latency | 842.25 ms |
| Evaluation elapsed time | 1,011.03 s (16 min 51 s) |
| Mean game runtime | 10.108 s |
| Throughput | 0.0989 games/s |

The Wilson 95% interval for 2048 and 4096 is **96.3–100.0%**; for 8192 it is **94.6–99.8%**; for 16384 it is **27.3–45.8%**. Thus 100/100 is an observed rate, not proof that the underlying success probability is 100%. The large maximum latency is an outlier and confirms that the 1 ms budget is a search stopping target rather than a hard deadline.

### Provenance and reproduction

- Checkpoint SHA-256: `d39f42fc3b7c59029a7a7d591c7cea58731983c0643d0420dce589599b929db5`
- Native library SHA-256: `1f5ef8b5e32016f7c615fff1d2f1035a490b34d879ba80cfb5d8b84f619af7d1`
- Raw JSON report SHA-256: `45326fec7b1bffc3201e651a5004c8ae07dd55a288df3ce6b50180fc7aca35e5`
- Source revision: `95377817b5dac58e703e58966abbf67c0afcbc2d`
- Runtime: WSL2 Ubuntu 24.04, Ryzen 5 5600X (6 cores / 12 threads), GCC 13.3.0, Python 3.12.3, NumPy 2.5.2.
- Native build flags: `-O3 -march=native -DNDEBUG -std=c++17 -fPIC -shared`.

Command:

```bash
python scripts/benchmark_hybrid.py \
  --games 100 \
  --seed 9100000 \
  --depth 3 \
  --node-budget 500000 \
  --time-budget-ms 1.0 \
  --ntuple models/ntuple_fresh_continue1m5_more500k.npz \
  --output reports/hybrid_latest_100_seed9100000.json
```

The report is retained locally and excluded from Git, as are the checkpoint and compiled native library. Reproduction requires obtaining the exact checkpoint separately and matching its SHA-256; this repository alone does not contain the 256 MiB model weights.

### Exploratory 10-game pilot on the same checkpoint

An earlier, separate pilot used seeds `9000000–9000009` and the same search configuration. It produced a mean score of 258,843.6 and a 70% rate for 16384. It is deliberately not combined with the 100-game holdout: the sample is small, the seed range is different, and selecting only the more favorable run would be misleading.

## Historical hybrid holdout: earlier 1M-continuation checkpoint, 100 games

| Metric | Result |
|---|---:|
| Seed range | 1,000,000–1,000,099 |
| Search depth cap | 3 |
| Node budget | 500,000 |
| Time budget | 1.0 ms/move |
| Mean score | 190,098.6 |
| Median score | 175,480 |
| Score range | 76,520–342,984 |
| Mean moves | 7,696.18 |
| Reached 2048 | 100% |
| Reached 4096 | 100% |
| Reached 8192 | 89% |
| Reached 16384 | 28% |
| Maximum tile | 16384 |
| Mean nodes/game | 12,648,959 |
| Mean cache hits/game | 2,010,867 |
| Mean completed depth | 2.055 |
| Mean decision latency | 1.027 ms |
| Decision latency P95 | 1.085 ms |
| Maximum recorded decision latency | 107.0 ms |

The observed 2048 rate was 100/100. Its Wilson 95% confidence interval is **96.3–100.0%**, so this run does not demonstrate a 99% lower confidence bound. The latency maximum is an outlier relative to the mean and P95; the time budget is not a hard execution deadline.

The checkpoint was the earlier local model referred to by the experiment name (`ntuple_fresh_continue1m`). It used the same search settings but a different checkpoint and seed range, so it is not a paired comparison with the latest run. The raw per-game report and checkpoint are local; their exact checkpoint hash and source revision were not recorded, so this historical result cannot be independently reconstructed from this repository alone.

## GUI evaluation: N-tuple + Expectimax, 100 games

| Metric | Result |
|---|---:|
| Seed range | 1,000,000–1,000,099 |
| Search depth cap | 3 |
| Node budget | 500,000 |
| Time budget | 1.0 ms/move |
| Mean score | 114,867.8 |
| Median score | 124,110 |
| Score range | 38,344–227,188 |
| Reached 2048 | 100% |
| Reached 4096 | 99% |
| Reached 8192 | 57% |
| Reached 16384 | 1% |
| Maximum tile | 16384 |
| Mean nodes/game | 4,460,555 |
| Mean cache hits/game | 643,035 |
| Mean completed depth | 1.724 |

This GUI report used the same seed range as the hybrid report above but its checkpoint provenance was not recorded in the JSON. Its results must not be treated as a direct paired comparison with the 190,098.6 mean-score run. Its observed 2048 rate was also 100/100 (Wilson 95% CI: 96.3–100.0%).

## Greedy N-tuple holdouts

| Checkpoint | Games | Seeds | Mean score | Median score | P(2048) | P(4096) | P(8192) | Max tile |
|---|---:|---|---:|---:|---:|---:|---:|
| Continued 1M episodes | 100 | 5,000,000–5,000,099 | 101,283.2 | 80,622 | 100% | 99% | 42% | 8192 |
| Fresh-start 200k | 100 | 5,000,000–5,000,099 | 60,541.7 | 60,548 | 94% | 64% | 6% | 8192 |
| Latest continued checkpoint | 100 | 8,000,000–8,000,099 | 129,241.3 | 133,286 | 99% | 97% | 70% | 16384 |

These results were collected on the same seed range, but the checkpoint selection and training process were not a preregistered experiment. They are development data.

## Root tile-downgrading paired comparison

The corrected experiment compares the baseline and root-only tile-downgrading variant on the same 20 seeds (2,000,000–2,000,019), with depth cap 3, 500,000 nodes, and 1 ms/move. Earlier tile-downgrading reports are invalid because the native backend used in those runs did not yet implement the transform.

| Metric | Baseline | Root downgrade |
|---|---:|---:|
| Mean score | 209,419.2 | 193,690.6 |
| Median score | 177,402 | 176,066 |
| P(2048) | 100% | 100% |
| P(4096) | 100% | 100% |
| P(8192) | 90% | 95% |
| P(16384) | 40% | 20% |

Paired scores: 10 wins / 10 losses (no ties) for the downgrade candidate. Mean paired score delta was **−15,728.6**; median delta was −9,000. The paired-bootstrap 95% interval was **−63,068 to +31,196**. The interval includes zero and the sample is small, so the experiment is inconclusive and does not justify enabling the transform by default.

## Benchmark limits

- Raw reports are local and excluded from the public repository; the values above are transcribed from the report summaries.
- Checkpoint hashes are unavailable for the historical runs; the latest hybrid holdout above records its hash.
- Runs on different checkpoints are not comparable even when seed ranges match.
- Latency summaries do not describe a strict worst-case deadline.
- Sample sizes are development-scale; larger independent holdouts are required for strong claims.
