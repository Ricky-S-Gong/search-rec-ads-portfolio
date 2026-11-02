# KuaiRand-1K ablation study

This package keeps the frozen train/validation protocol unchanged. It does not read
`final-test-sealed` and it does not claim online A/B-test results.

## What is included

- `configs/ablation_control.yaml`: the matched 20-epoch attention + residual +
  BatchNorm + logQ control.
- `configs/ablation_plan.yaml`: architecture, one-factor hyperparameter, and
  sampling-correction variants.
- `run_ablation.py`: writes a resolved config for every run and launches training
  only with explicit `--execute`.
- `summarize_ablation.py`: collects completed run folders into CSV and Markdown.

## Colab workflow

After extracting this source bundle and mounting the existing 1K **train/validation**
data release, first validate the environment and materialize configs without training:

```bash
uv sync --frozen --python 3.11.16
uv run python run_ablation.py --data DATA_ROOT --output-root RUNS/ablation --group architecture
```

The command above must print `PLAN`, not `RUN`. Inspect the generated configs, then
launch one group at a time on CUDA:

```bash
uv run python run_ablation.py --data DATA_ROOT --output-root RUNS/ablation --group architecture --execute
uv run python run_ablation.py --data DATA_ROOT --output-root RUNS/ablation --group hyperparameters --execute
uv run python run_ablation.py --data DATA_ROOT --output-root RUNS/ablation --group sampling --execute
uv run python summarize_ablation.py --runs RUNS/ablation --output RUNS/ablation-summary
```

Use seed `42` for screening. After selecting the winner from each group, rerun only
the relevant control and winner with three seeds; do not rerun the whole matrix:

```bash
uv run python run_ablation.py --data DATA_ROOT --output-root RUNS/ablation-final --group sampling --seeds 42,2026,3407 --execute
```

That example is for the logQ comparison. For a final architecture or hyperparameter
comparison, make a small copy of `ablation_plan.yaml` containing only the control and
the chosen variant. Keep selection on validation; run the sealed test only once after
the final configuration is frozen.

## Matrix

| Group | Variants | Question |
| --- | --- | --- |
| Architecture | shallow mean; deep mean; deep attention; + residual; control + BatchNorm; LSTM replacing attention | Which model component adds value? |
| Hyperparameters | learning rate, dropout, weight decay, temperature | Which training choice is best, holding architecture fixed? |
| Sampling | none, 0.5 logQ, standard logQ, Laplace-smoothed logQ | Does correcting target-frequency sampling help? |

The control has standard logQ (`alpha=1`, `smoothing=0`). A logQ correction is

\[
\mathrm{logit}(u,i)=\cos(u,i)/\tau-\alpha\log Q(i).
\]

For the Laplace run, \(Q(i)=(count_i+1)/(N+K)\), where \(N\) is the count of
training positives and \(K\) is the number of distinct training target videos.
