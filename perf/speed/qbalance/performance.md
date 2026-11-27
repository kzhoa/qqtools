# Qbalance performance

Run from the repository root.

```bash
PYTHONPATH=src python perf/speed/qbalance/test_qbalance.py
```

The script calls the [production planner](../../../src/qqtools/data/qbalance.py)
for each LPT tier. It reports median CPU planning time over seeds 7, 13, and 29,
plus estimated batch and step loads at seed 7. Input generation and scoring
are outside the timed section.

## CPU planning

November 27, 2026 run on Linux 5.15, Intel Xeon Platinum 8480+, Python 3.13.15,
and NumPy 2.5.3. The cases below use rounded-lognormal costs, batch size 16,
and four ranks.

| Samples | Strategy | Planning ms | Peak batch cost | P99 batch cost | Sum of step maxima |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16,384 | `lpt_fast` | 4.59 | 10,397 | 2,652.13 | 259,166 |
| 16,384 | `lpt_medium` | 23.60 | 10,167 | 2,436.13 | 259,081 |
| 16,384 | `lpt_best` | 83.35 | 10,167 | 2,436.13 | 259,081 |

These measurements cover CPU initialization with synthetic sample costs.
[Earlier CPU timings](../../../docs/spec/balanced-sampler.md#two-pointer-production-integration-2026-09-05)

## Historical exact results

Across 300 small configurations, fast, medium, and best reached the exact
peak in 226, 267, and 299 cases. Mean peak gaps were 0.3282%, 0.1360%, and
0.0015%. The sampler specification records the exhaustive comparison and
its reproduction context.
[Exact results](../../../docs/spec/balanced-sampler.md#stage-split-verification-2026-09-05)
