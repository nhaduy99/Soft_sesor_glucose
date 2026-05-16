# TODO

## Highest priority
- Merge the quantitative HPLC Rhamnose target table into `eem_raman_hplc_inventory_enriched.csv`.
- Verify exact target units and whether the prediction target should be concentration, peak area, or another HPLC-derived value.
- Decide the primary supervised cohort:
  - Raman-only rows
  - EEM-only rows
  - Paired EEM+Raman rows

## Data quality
- Recover the missing raw CSV files listed in `___All_Errors.txt`.
- Recover the missing flask metadata workbook if available.
- Confirm whether any EEM wavelength regions should be hard-masked beyond the current `OVER` and near-diagonal rules.

## Modeling
- Create a train/test-ready table with numeric targets.
- Train baseline PLS models on:
  - Raman interpretable features
  - Raman full vectors
  - EEM cleaned hotspot features
  - EEM full cleaned vectors
  - Fusion features
- Compare grouped splits by `batch` versus `metadata_experiment`.

## Reporting
- Add supervised evaluation plots after targets are available:
  - predicted vs actual
  - residual distribution
  - error by batch / experiment
- Add explicit scientific interpretation notes for the most influential Raman and EEM features from supervised loadings.
