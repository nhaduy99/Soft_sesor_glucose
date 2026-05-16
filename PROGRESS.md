# Project Progress

Last updated: 2026-05-17

## Current goal
Prepare a scientifically defensible, ML-ready workflow for predicting Rhamnose from EEM and Raman data once quantitative HPLC targets are available.

## Current status
Raw data have been inventoried and matched across EEM and Raman using filename structure plus plate metadata. An enriched inventory exists and includes experiment, plate, well, treatment labels, and HPLC sample-code mapping from the available legend workbooks. Scientific visualizations have been generated for Raman and EEM, including a second-pass cleaned EEM analysis that masks saturated and near-diagonal scatter regions. Interpretable and full feature tables have been exported for ML use, unsupervised PCA/K-means exploration has been completed on the interpretable features, and supervised standard/spike-based monosaccharide soft-sensor models have been trained.

The main missing piece for culture-sample prediction is still the quantitative HPLC monosaccharide reference table. Standards and known spikes now support supervised calibration experiments, but culture rows without quantitative targets remain excluded from final supervised evaluation.

## Recently completed
- Built raw and enriched inventory tables covering EEM/Raman measurement availability and metadata joins.
- Generated annotated Raman and EEM visual reports in SVG and HTML.
- Exported ML-ready feature tables from cleaned EEM and cropped Raman data.
- Ran unsupervised PCA and clustering on interpretable features and saved an HTML report.
- Created a starter `rhamnose_ml` training scaffold for future supervised modeling.
- Added `docs/softsensor_monosaccharide_model_io.html`, a self-contained HTML explanation of model inputs, outputs, candidate pipelines, and the recommended Raman + EEM mid-level fusion strategy for monosaccharide prediction.
- Added `train_monosaccharide_softsensor.py` and generated supervised standard/spike calibration outputs in `supervised_monosaccharides/`. The iterative search improved RMSE versus the initial linear/no-log/all-known baseline by 16.4% for rhamnose, 11.7% for xylose, and 22.7% for glucose.

## Important files
- `build_enriched_inventory.py`: builds the enriched sample inventory by joining raw file structure with metadata and HPLC sample legends.
- `eem_raman_hplc_inventory_enriched.csv`: master enriched inventory used as the central join table.
- `visualize_eem_raman.py`: aggregates all raw EEM/Raman files, creates annotated figures, and writes the scientific HTML report.
- `visualizations/visual_report.html`: integrated spectroscopy analysis report with inline figures.
- `export_rhamnose_features.py`: exports interpretable and full ML feature tables from raw spectroscopy files.
- `features/rhamnose_interpretable_features.csv`: compact feature set for exploratory and baseline modeling.
- `features/rhamnose_full_feature_matrix.csv`: high-dimensional feature matrix for full-spectrum / full-matrix models.
- `explore_features_unsupervised.py`: runs PCA and K-means using `numpy` only.
- `unsupervised/unsupervised_report.html`: integrated unsupervised exploration report.
- `docs/softsensor_monosaccharide_model_io.html`: model input/output explanation with diagrams for monosaccharide soft-sensor prediction.
- `train_monosaccharide_softsensor.py`: pure-NumPy supervised training and model-search script for rhamnose, xylose, and glucose standards/spikes.
- `supervised_monosaccharides/supervised_report.html`: supervised model results report with target coverage, training flow, best models, and optimization improvement.
- `supervised_monosaccharides/optimization_improvement_summary.csv`: final-best RMSE compared with the initial baseline search.
- `rhamnose_ml/scripts/train_baseline.py`: starter baseline training entry point for supervised Rhamnose prediction.
- `rhamnose_ml/src/rhamnose_ml/train.py`: baseline training pipeline using PLS with grouped train/test splitting.

## Current blockers / bugs
- No quantitative HPLC monosaccharide target file is currently available for culture-sample supervised training.
- `scikit-learn`, `pandas`, and `matplotlib` are not installed in the current environment, so advanced analysis was implemented with lower-level tooling.
- Some directory listings in PowerShell returned stale snapshots; direct file existence checks were used as fallback validation.

## Next recommended steps
1. Obtain and merge the actual quantitative HPLC monosaccharide reference table for culture samples.
2. Re-run supervised training with culture targets and compare against the current standards/spikes calibration results.
3. Add stronger Raman baseline correction and EEM scatter masking, then rerun `train_monosaccharide_softsensor.py`.
4. If external dependencies are allowed, compare the pure-NumPy search against scikit-learn PLSR/SVR/XGBoost implementations.

## How to run
```bash
pip install -r rhamnose_ml/requirements.txt
python build_enriched_inventory.py
python visualize_eem_raman.py
python export_rhamnose_features.py
python explore_features_unsupervised.py
python train_monosaccharide_softsensor.py
python rhamnose_ml/scripts/train_baseline.py --config rhamnose_ml/config/defaults.json
```
