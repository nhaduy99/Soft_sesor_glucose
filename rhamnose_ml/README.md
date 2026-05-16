# Rhamnose ML Starter

This project is a starter scaffold for predicting **Rhamnose** from paired `EEM` and `Raman` measurements.

It is designed around the enriched inventory already generated in `Codex_inventory`, with one important constraint:

- the current dataset does **not** yet contain a quantitative HPLC Rhamnose target table
- the training code expects a numeric target column to be added to the master inventory before model fitting

## Project Layout

```text
rhamnose_ml/
  config/
    defaults.json
  data/
    README.md
  docs/
    training_pipeline.md
  outputs/
    README.md
  scripts/
    train_baseline.py
  src/
    rhamnose_ml/
      __init__.py
      config.py
      io.py
      features.py
      train.py
  requirements.txt
  README.md
```

## Inputs

The baseline pipeline expects:

1. An enriched inventory CSV
   - current default:
   - `..\eem_raman_hplc_inventory_enriched.csv`
2. Raw EEM CSV files
3. Raw Raman CSV files
4. A numeric Rhamnose target column added to the inventory

Default target column in config:

- `target_rhamnose`

If your actual HPLC table uses another column name, update `config/defaults.json`.

## Model Strategy

The starter pipeline trains a **PLS regression** baseline on:

- `EEM-only`
- `Raman-only`
- `EEM + Raman` fused features

This is a strong first model for spectroscopy because it handles high-dimensional, correlated signals well.

## Quick Start

1. Add a numeric Rhamnose target column to the enriched inventory.
2. Review `config/defaults.json`.
3. Run:

```powershell
python scripts/train_baseline.py --config config/defaults.json
```

## Outputs

The baseline script writes:

- model metrics CSV
- sample-level predictions CSV
- fitted pipeline objects as `joblib`

into `outputs/`.

## Current Limitation

At this stage the scaffold is ready, but training cannot produce meaningful Rhamnose predictions until the quantitative HPLC Rhamnose reference values are added to the enriched inventory.
