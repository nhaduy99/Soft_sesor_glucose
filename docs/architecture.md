# Architecture Notes

## Purpose
This workspace acts as a thin analysis and modeling layer on top of Emilie's raw spectroscopy data and metadata workbooks.

## Core flow

```text
Raw EEM/Raman CSVs
        +
Metadata / legends
        |
        v
Matched inventory
        |
        +--> visual diagnostics
        |
        +--> feature export
        |
        +--> unsupervised exploration
        |
        +--> supervised Rhamnose modeling
```

## Main components

### 1. Inventory layer
- `build_enriched_inventory.py`
- outputs:
  - `eem_raman_hplc_inventory.csv`
  - `eem_raman_hplc_inventory_enriched.csv`

This is the primary join layer. It maps raw files to metadata, experiment layout, and HPLC sample legend codes.

### 2. Visualization layer
- `visualize_eem_raman.py`
- outputs in `visualizations/`

This layer provides scientific diagnostics:
- Raman mean spectrum and heatmap
- raw EEM mean map
- cleaned EEM mean map
- EEM saturation map

### 3. Feature layer
- `export_rhamnose_features.py`
- outputs in `features/`

Two feature products are created:
- interpretable low-dimensional features
- full high-dimensional feature matrix

### 4. Unsupervised analysis layer
- `explore_features_unsupervised.py`
- outputs in `unsupervised/`

This layer helps assess structure before supervised modeling:
- PCA projections
- K-means clusters
- PC1 loading summaries

### 5. Supervised modeling scaffold
- `rhamnose_ml/`

This is the future supervised training package. It is ready to train once numeric HPLC Rhamnose targets are available.

## Design assumptions
- Raw file naming is stable and meaningful.
- Plate metadata are currently more complete than flask metadata.
- `OVER` values and near-diagonal EEM regions should be treated cautiously.
- Raman information between `500-2000 cm^-1` is the primary predictive region.
- Quantitative HPLC targets will eventually be merged into the enriched inventory rather than handled as a separate disconnected table.

## Current limitation
No quantitative HPLC Rhamnose target table is available yet. That is the main blocker for supervised prediction.
