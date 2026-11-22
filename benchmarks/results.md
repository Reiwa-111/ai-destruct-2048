# Benchmark results

These are selected development measurements. They are not state-of-the-art claims. The repository currently does not include trained checkpoints, and several historical reports do not record the exact checkpoint hash; treat those results as informative but not fully reproducible.

## Hybrid: continued N-tuple checkpoint, 100 games

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

The checkpoint is the local continued N-tuple model referred to by the experiment name (`ntuple_fresh_continue1m`). This seed range differs from the training-time greedy validation ranges recorded for the 1M and 1.5M continuation runs (6,000,000 and 7,000,000, respectively). The raw per-game report is retained locally as `reports/hybrid_fresh_continue1m_100_seed1000000.json`, but raw reports and model files are excluded from Git. The checkpoint hash and exact source revision were not recorded in the report; this result therefore cannot be independently reconstructed from this repository alone.

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
- Checkpoint hashes are unavailable for these historical runs.
- Runs on different checkpoints are not comparable even when seed ranges match.
- Latency summaries do not describe a strict worst-case deadline.
- Sample sizes are development-scale; larger independent holdouts are required for strong claims.
