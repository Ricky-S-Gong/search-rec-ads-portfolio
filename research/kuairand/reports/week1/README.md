# Week 1 English technical report

`technical_report_draft.tex` and its PDF are a measured **draft**, not a completed Stage 1 certificate. `report_evidence.json` and `evidence/` preserve the aggregate input facts. Corrected basic/enhanced CUDA results and matching interruption/recovery evidence must arrive before finalization.

The standalone LaTeX source includes EDA plot coordinates, actual field examples and both tower diagrams. It can compile without external figures or BibTeX files. The generator writes a final filename only for a complete stage; otherwise `--allow-incomplete` is required and the draft label is visible throughout.

## Rebuild from a completed private release

Run from the repository root with Python 3.11 or later and a local `pdflatex` installation. The variables below point to already mounted private artifacts.

```bash
python research/kuairand/reports/week1/build_report.py \
  --audit "$DATA_DIR/audit.json" \
  --manifest "$DATA_DIR/manifest.json" \
  --stage "$RELEASE_DIR/stage_report.json" \
  --runs "$RUN_DIR" \
  --output research/kuairand/reports/week1 \
  --compile
```

Optionally pass `--historical /path/to/unpacked/historical-b2ed5728b9a2` to include the original loss curve. The adjacent `historical-evaluation.json` is used only as a clearly labeled diagnostic. Preserve historical model/config/manifest/frozen source hashes; never rewrite these to match the new package.

`evaluate_historical.py` reproduces diagnostic inference using the original source ZIP's `kuairand` package, with matching PyTorch base version and strict original code/config/data/vector/checkpoint checks. Its CPU inference does not resume or certify original CUDA training.

Compilation uses a temporary directory that is automatically removed, retaining editable `.tex`, PDF and lightweight evidence. Finalization must visually inspect every page and verify tables against aggregate JSON. The sealed final test bundle is not a report input.
