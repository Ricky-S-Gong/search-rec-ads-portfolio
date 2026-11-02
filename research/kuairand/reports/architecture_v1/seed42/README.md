# Architecture ablation: seed 42

**Status: six completed validation runs; single-seed exploratory results. Seeds 2026 and 3407 are not yet complete. These are not sealed-test results.**

| ID | MLP | History | Best epoch | Recall@100 | NDCG@100 | Hits |
|---|---|---|---:|---:|---:|---:|
| B0 | shallow | mean | 10 | 0.0010329742 | 0.0002119553 | 574 |
| A5 | deep | mean | 8 | 0.0010995596 | 0.0002307275 | 611 |
| R0 | deep | attention | 7 | 0.0011985380 | 0.0002497747 | 666 |
| A1 | deep | lstm | 7 | 0.0012021372 | 0.0002584420 | 668 |
| A2 | shallow | attention | 14 | 0.0011535478 | 0.0002574765 | 641 |
| A3 | shallow | lstm | 9 | 0.0011373514 | 0.0002350446 | 632 |

Metrics are on the 0–1 scale. NDCG is reported at the checkpoint selected by Recall first, with NDCG as a tie-breaker; it is not an independently maximized NDCG.

## Protocol and interpretation

- Same source hash, frozen data manifest, seed, training settings and evaluation protocol across all six runs. No batch normalization, residual blocks, dropout or sampling correction.
- Adam, learning rate 0.001, batch size 256, temperature 0.1; at most 20 epochs, early stopping after 3 non-improving validation epochs.
- 1,335,793 training queries; 555,677 validation queries; catalog contains 4,371,868 items. Candidate eligibility follows `upload_next_day`; ranking is exhaustive FP32 with ascending raw-ID tie-breaks, and repeat recommendations are allowed.
- Target coverage is 0.7573842358. Uncovered targets remain in the denominator and count as misses. This limitation must accompany the reported metrics.
- B0 used T4; A5/R0/A1/A2 acceptance records show A100; A3 final acceptance shows L4 after recovery. Do not treat mixed-device runtimes as architecture-only speed comparisons.
- A3 has evaluation evidence for epochs 1–12, matching final step 62,616 and sample count 16,029,516. All 12 epoch artifact receipts report synced. This checks recorded continuity, not a bitwise comparison with an uninterrupted run.
- A1 and R0 differ by only two validation hits (668 versus 666). No stable superiority or statistical significance is established from one seed.
- Depth improves Mean on both metrics here; deeper Attention improves Recall but reduces NDCG relative to shallow Attention.

## Files

- [results.csv](results.csv): one row per configuration.
- [epoch_metrics.csv](epoch_metrics.csv): complete per-epoch metrics and timings.
- `evidence/`: resolved configs, run manifests, per-epoch validation records and W&B artifact receipts. Checkpoint weights remain in Drive/W&B; receipts do not contain weights.
- [Training source and reproduction guide](../../../training/architecture_v1/ARCHIVE_GUIDE.md).

## Loss and metric formulas

The no-correction training loss is in-batch cross entropy. For user embedding u_i and item embedding v_j, s_ij = dot(u_i, v_j) / temperature. The positive is the diagonal pair; off-diagonal duplicates of the positive item ID are masked. L = -(1/B) sum_i log[exp(s_ii) / sum_(j in unmasked batch items) exp(s_ij)]. See `training/kt1k/model.py::loss_fn` for the exact implementation.

Each validation query has one target. Recall@100 = sum_i 1(rank_i <= 100) / N. NDCG@100 = sum_i [1(rank_i <= 100) / log2(rank_i + 1)] / N, using 1-based ranks; uncovered targets contribute zero. Validation loss is a separate fixed-order, batch-256, FP32 in-batch loss, not full-catalog cross entropy.

## Next stage

Keep all seed42 results. Add seeds 2026 and 3407 for each unique configuration (12 runs), then report paired seed differences, means and sample standard deviations. A4 reuses B0 and is not a seventh independent configuration.
