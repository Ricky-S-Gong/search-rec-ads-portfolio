# KuaiRand recommendation project / KuaiRand 推荐项目

The project separates data preparation, model training, evaluation and reports:

| Directory | Responsibility | Entry point |
| --- | --- | --- |
| `data_pipeline/` | Official logs, temporal splits, histories and static features | [Data guide](data_pipeline/README.md) |
| `training/` | Two towers, training, checkpoints, W&B and vector export | [Training guide](training/README.md) |
| `evaluation/` | Frozen validation protocol, exact retrieval, recovery evidence and stage release | `training/run.py evaluate` |
| `reports/week1/` | English LaTeX technical report and generated figures | Report generation script |
| `training/recommender/` | Python implementation of the model and trainer | `training/run.py` |
| `training/recommender/vendor/ziqi/` | Immutable upstream snapshot and provenance | Read-only; retain original hashes |

数据流水线和模型训练分别放在 `data_pipeline/` 与 `training/`。前者使用仓库根目录的研究环境；后者使用自己的 `pyproject.toml` 和 `uv.lock`，由 Colab 手动启动。测试与配置放在所属执行环境旁边。

Git stores source, tests, configuration and documentation. Drive stores datasets, checkpoints and vectors. W&B stores experiment logs. The website under `src/` consumes only compact public artifacts under `public/artifacts/`.

## Commands / 运行入口

From the repository root:

```bash
uv run python research/kuairand/data_pipeline/prepare_kuairand_data.py --help
uv run --directory research/kuairand/training python run.py check-config
uv run --directory research/kuairand/training pytest tests -q
```

Use [the Colab notebook](training/notebooks/week1_baseline.ipynb) for GPU training. For a PR checkout, set its `REF` to the PR branch; after merge use `main`, or pin an exact `COMMIT`.

## Path migration / 路径迁移

| Previous path | Current path |
| --- | --- |
| `research/kuairand-twotower/` | `research/kuairand/data_pipeline/` |
| `kuairand_twotower` Python package | `kuairand_data` |
| `research/kuairand/run.py` and training environment | `research/kuairand/training/` |
| `research/kuairand/kuairand/` Python package | `research/kuairand/training/recommender/` |

Existing Drive locations, dataset hashes and W&B run IDs remain the historical evidence. The frozen vendor snapshot retains original paths in its provenance documentation; those are historical references, not current commands.

The package/factory name has changed, so new configurations have a different config hash. Resume historical runs using the exact source ZIP, configuration and prepared manifest in the shared model bundle, or their original Git commit. Do not edit old manifests or bypass checkpoint binding checks. New runs use the current layout and new output directories.

旧实验使用其冻结源码和原配置恢复；新协议启动独立实验。使用 week1_baseline.ipynb 读取挂载 Drive 的已发布数据，运行基础／历史双塔及精确评估。只有 stage_report.json 的 complete=true 才表示第一周验收完成。
