# Project Progress

Last updated: 2026-05-12

## Current goal
Prepare a scientifically defensible, ML-ready workflow for predicting Rhamnose from EEM and Raman data once quantitative HPLC targets are available.

## Current status
Raw data have been inventoried and matched across EEM and Raman using filename structure plus plate metadata. An enriched inventory exists and includes experiment, plate, well, treatment labels, and HPLC sample-code mapping from the available legend workbooks. Scientific visualizations have been generated for Raman and EEM, including a second-pass cleaned EEM analysis that masks saturated and near-diagonal scatter regions. Interpretable and full feature tables have been exported for ML use, and unsupervised PCA/K-means exploration has been completed on the interpretable features.

The main missing piece is still the quantitative HPLC Rhamnose reference table. Without that target table, supervised model training remains scaffold-only.

## Recently completed
- Built raw and enriched inventory tables covering EEM/Raman measurement availability and metadata joins.
- Generated annotated Raman and EEM visual reports in SVG and HTML.
- Exported ML-ready feature tables from cleaned EEM and cropped Raman data.
- Ran unsupervised PCA and clustering on interpretable features and saved an HTML report.
- Created a starter `rhamnose_ml` training scaffold for future supervised modeling.

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
- `rhamnose_ml/scripts/train_baseline.py`: starter baseline training entry point for supervised Rhamnose prediction.
- `rhamnose_ml/src/rhamnose_ml/train.py`: baseline training pipeline using PLS with grouped train/test splitting.

## Current blockers / bugs
- No quantitative HPLC Rhamnose target file is currently available in the working dataset.
- Flask metadata workbook is still missing locally, so flask rows are not as richly annotated as plate rows.
- `scikit-learn`, `pandas`, and `matplotlib` are not installed in the current environment, so advanced analysis was implemented with lower-level tooling.
- Some directory listings in PowerShell returned stale snapshots; direct file existence checks were used as fallback validation.

## Next recommended steps
1. Obtain and merge the actual quantitative HPLC Rhamnose reference table into the enriched inventory.
2. Create a supervised training table restricted to rows with valid targets and the desired modality set (`Raman`, `EEM`, or `Fusion`).
3. Train and compare baseline `PLS` models for Raman-only, EEM-only, and fused features.
4. Evaluate whether cleaned EEM features improve prediction beyond Raman alone.
5. If useful, add stronger preprocessing for Raman baseline correction and EEM masking/scatter removal.

## How to run
```bash
pip install -r rhamnose_ml/requirements.txt
python build_enriched_inventory.py
python visualize_eem_raman.py
python export_rhamnose_features.py
python explore_features_unsupervised.py
python rhamnose_ml/scripts/train_baseline.py --config rhamnose_ml/config/defaults.json
```
