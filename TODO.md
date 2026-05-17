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
- Refine the already-added spectroscopy preprocessing before another optimization attempt:
  - Raman cosmic-spike removal, ALS baseline correction, smoothing/derivatives, SNV, and optional area normalization are implemented in `preprocessing_raman.py`.
  - EEM `OVER` handling, near-diagonal scatter masking, and PARAFAC score export are implemented in `eem_parafac_features.py`.
  - replicate-aware calibration still needs quantitative HPLC culture targets.
- Current focused preprocessed/PARAFAC search met a 5% improvement only for glucose. Rhamnose and xylose remain unresolved with the current standard/spike labels.
- The `Rha (5)` exclusion sensitivity run improved rhamnose and xylose RMSE relative to the previous report but changed the best rhamnose model from Raman+EEM fusion to EEM-only ridge. Treat this as evidence that the 5 g/L rhamnose standards strongly influenced the earlier fusion result.
- Improve PARAFAC numerical quality and validation before relying on PARAFAC scores for final claims:
  - add non-negativity/stability constraints with a tested library if dependencies are available
  - evaluate component interpretability against known fluorophores/process-state signals
  - rerun rank selection after quantitative HPLC culture targets are merged

## Reporting
- Review `supervised_monosaccharides/monosaccharide_softsensor_comprehensive_report.docx` in Word and refine wording/figures for the intended audience if it will be used as a thesis or publication appendix.
- Review `supervised_monosaccharides_exclude_rha5/monosaccharide_softsensor_exclude_rha5_report.docx` as the sensitivity-analysis report for excluding the 5 g/L rhamnose standards.
- Add supervised culture-sample evaluation plots after HPLC targets are available:
  - predicted vs actual
  - residual distribution
  - error by batch / experiment
- Add explicit scientific interpretation notes for the most influential Raman and EEM features from supervised loadings and selected kNN features.
- Extend `supervised_monosaccharides/comprehensive_modeling_report.html` after culture-target training so it separates standards, spikes, and true culture samples.
- Add external reference assignments for Raman bands if a publication-ready spectral interpretation section is required.
