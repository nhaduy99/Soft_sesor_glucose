# TODO

## Highest priority
- Merge the quantitative HPLC monosaccharide target table into `eem_raman_hplc_inventory_enriched.csv`.
- Verify exact target units and whether the prediction target should be concentration, peak area, or another HPLC-derived value.
- Re-run `train_monosaccharide_softsensor.py` after adding culture-sample targets.

## Data quality
- Recover the missing raw CSV files listed in `___All_Errors.txt`.
- Recover the missing flask metadata workbook if available.
- Confirm whether any EEM wavelength regions should be hard-masked beyond the current `OVER` and near-diagonal rules.

## Modeling
- Extend the current standard/spike supervised training to quantitative HPLC culture targets when available.
- Treat the additional 20% RMSE-improvement target as unresolved with the current standard/spike labels. Latest extra gains over the previous best are rhamnose 2.4%, xylose 0.4%, and glucose 5.9%.
- Compare the current pure-NumPy Ridge/PCR/PLS/kNN search with dependency-backed models if package installation is allowed:
  - scikit-learn PLSR
  - SVR
  - XGBoost or histogram gradient boosting
- Compare grouped splits by `batch` versus `metadata_experiment`.
- Add true spectroscopy preprocessing before another optimization attempt:
  - Raman baseline correction / smoothing
  - EEM scatter-region masking and PARAFAC scores
  - replicate-aware calibration using quantitative HPLC targets

## Reporting
- Add supervised culture-sample evaluation plots after HPLC targets are available:
  - predicted vs actual
  - residual distribution
  - error by batch / experiment
- Add explicit scientific interpretation notes for the most influential Raman and EEM features from supervised loadings and selected kNN features.
- Extend `supervised_monosaccharides/comprehensive_modeling_report.html` after culture-target training so it separates standards, spikes, and true culture samples.
