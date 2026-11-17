# Predicting speed-dating compatibility (MixMoSa dataset)

Nested, event-grouped comparison of 8 models (+ a prior reference) that predict whether
two people will **both** say "yes" after an 8-minute speed date, using only information
available before the date. Full write-up: [`REPORT.md`](REPORT.md).

## Layout

| Path | Content |
|---|---|
| `src/features.py` | Loads the raw CSVs, builds the pair-level table (84 features) -> `data_processed/pairs.csv` |
| `src/models.py` | Model zoo (sklearn, LightGBM, PyTorch MLP, multi-task two-tower net) + Optuna search spaces |
| `src/metrics.py` | ROC-AUC, PR-AUC, Brier, log-loss, precision/lift in top 10 % |
| `src/train_nested_cv.py` | Nested CV: outer 5-fold x 2 repeats grouped by event, inner 3-fold Optuna tuning |
| `src/analysis.py` | EDA, bootstrap CIs, learning / training / complexity curves, permutation importance |
| `results/` | Out-of-fold predictions, per-fold metrics, tuned parameters, `model_comparison.csv`, `summary.json` |
| `src/make_pdf.py` | Renders `REPORT.md` to `REPORT.pdf` |
| `figures/` | All plots used in the report |

## Reproduce (Python 3.12)

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
git clone --depth 1 https://github.com/dankras/mixmosa-anonymized-data.git data_raw/mixmosa
.venv/bin/python src/features.py            # ~2 s
.venv/bin/python src/train_nested_cv.py     # ~25 min on 10 CPU cores
.venv/bin/python src/analysis.py            # ~5 min
.venv/bin/python src/make_pdf.py            # REPORT.md -> REPORT.pdf (headless google-chrome)
```

`train_nested_cv.py --models LightGBM MLP --repeats 1 --jobs 4` runs a subset.
