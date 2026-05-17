# Project Progress

Last updated: 2026-05-17

## Current goal
Prepare a scientifically defensible, ML-ready workflow for predicting Rhamnose from EEM and Raman data once quantitative HPLC targets are available.

## Current status
Raw data have been inventoried and matched across EEM and Raman using filename structure plus plate metadata. An enriched inventory exists and includes experiment, plate, well, treatment labels, and HPLC sample-code mapping from the available legend workbooks. Scientific visualizations have been generated for Raman and EEM, including a second-pass cleaned EEM analysis that masks saturated and near-diagonal scatter regions. Interpretable and full feature tables have been exported for ML use, unsupervised PCA/K-means exploration has been completed on the interpretable features, and supervised standard/spike-based monosaccharide soft-sensor models have been trained and further optimized with distance-normalized kNN and kernel-ridge variants.

The main missing piece for culture-sample prediction is still the quantitative HPLC monosaccharide reference table. Standards and known spikes now support supervised calibration experiments, but culture rows without quantitative targets remain excluded from final supervised evaluation.

## Recently completed
- Built raw and enriched inventory tables covering EEM/Raman measurement availability and metadata joins.
- Generated annotated Raman and EEM visual reports in SVG and HTML.
- Exported ML-ready feature tables from cleaned EEM and cropped Raman data.
- Ran unsupervised PCA and clustering on interpretable features and saved an HTML report.
- Created a starter `rhamnose_ml` training scaffold for future supervised modeling.
- Added `docs/softsensor_monosaccharide_model_io.html`, a self-contained HTML explanation of model inputs, outputs, candidate pipelines, and the recommended Raman + EEM mid-level fusion strategy for monosaccharide prediction.
- Added `train_monosaccharide_softsensor.py` and generated supervised standard/spike calibration outputs in `supervised_monosaccharides/`. The iterative search improved RMSE versus the initial linear/no-log/all-known baseline by 16.4% for rhamnose, 11.7% for xylose, and 22.7% for glucose.
- Ran a further optimization pass with cosine/correlation/Manhattan kNN, row-wise normalization, and focused RBF/Laplacian kernel ridge. Latest confirmed results improved the previous best by 2.4% for rhamnose, 0.4% for xylose, and 5.9% for glucose; the requested additional 20% improvement was not reached with the current standard/spike labels.
- Added `generate_supervised_visual_report.py` and generated `supervised_monosaccharides/comprehensive_modeling_report.html` with predicted-vs-true scatter plots, residual plots, RMSE/improvement bar charts, top-model tables, and an end-to-end modelling pipeline summary.
- Extended the comprehensive report with processed spectroscopy examples: annotated Raman overlays for monosaccharide standards/mixtures, EEM heatmaps for rhamnose/xylose/glucose examples, saturation/scatter annotations, and written image analysis explaining direct Raman sugar signals versus indirect EEM process-state signals.
- Added `preprocessing_raman.py` to generate Raman features with cosmic spike removal, asymmetric least-squares baseline correction, Savitzky-Golay smoothing, first/second derivatives, SNV normalization, optional area normalization, and a `preprocessing_config` column for model traceability.
- Added `eem_parafac_features.py` to load the cleaned EEM cube, fit PARAFAC ranks 2-8, score ranks by reconstruction error, split-half stability, and prediction performance, export selected-rank scores/loadings, and create loading/component-map SVG plots.
- Ran Raman preprocessing and EEM PARAFAC exports. `features/raman_preprocessed_features.csv` contains 765 labelled/configured Raman rows. PARAFAC selected rank 2 and exported scores/loadings/component maps under `features/eem_parafac/`.
- Added `train_preprocessed_models.py` to compare Raman-preprocessed, PARAFAC-score, and PARAFAC+Raman-fusion models against the last best RMSEs. Glucose improved by 11.6% versus the last best model; rhamnose and xylose did not improve with the new features.
- Added kernel-ridge candidates to the focused preprocessed/PARAFAC search and regenerated the comprehensive HTML report with an explicit feature input/output table for modelling. Glucose improved further to 0.5094 RMSE, an 8.8% improvement versus the latest project-level baseline. Rhamnose and xylose still did not meet the requested 5% improvement threshold with the current standard/spike labels.
- Added `generate_docx_model_report.py` and generated `supervised_monosaccharides/monosaccharide_softsensor_comprehensive_report.docx`, a 6-8 page Word report with pipeline diagrams, result plots, PARAFAC visual demonstrations, strengths, weaknesses, biological interpretation, and next-step recommendations.
- Added an exclusion mode for all `Rha (5)` examples, reran supervised model search and preprocessing/PARAFAC comparison, and generated a separate report set under `supervised_monosaccharides_exclude_rha5/`. The filtered run contains zero rows with `rhamnose_gL = 5`.

## Latest supervised results
| Target | Best cohort | Best feature set | Best model | RMSE | Improvement vs initial baseline | Additional improvement vs previous best |
|---|---|---|---|---:|---:|---:|
| Rhamnose | target_focused | fusion_full | kNN, Manhattan, L2 row normalization | 0.7043 | 18.4% | 2.4% |
| Xylose | all_known | eem_full | Laplacian kernel ridge, log target | 0.5359 | 12.0% | 0.4% |
| Glucose | target_focused | eem_interpretable | kNN, Manhattan | 0.5589 | 27.3% | 5.9% |

## Latest preprocessing/PARAFAC model results
| Target | Best new feature set | Best model | RMSE | Latest project baseline RMSE | Improvement vs latest baseline | Met 5% |
|---|---|---|---:|---:|---:|---|
| Rhamnose | parafac_raman_fusion_als_sg0_snv_area | weighted kNN | 0.8001 | 0.7043 | -13.6% | No |
| Xylose | raman_preprocessed_als_sg2_snv | weighted kNN | 0.6050 | 0.5359 | -12.9% | No |
| Glucose | raman_preprocessed_als_sg2_snv | Laplacian kernel ridge | 0.5094 | 0.5589 | 8.8% | Yes |

## Latest `Rha (5)` exclusion results
| Target | Previous best RMSE | New filtered best RMSE | Change vs previous | New filtered best model |
|---|---:|---:|---:|---|
| Rhamnose | 0.7043 | 0.4136 | 41.3% better | EEM full + ridge |
| Xylose | 0.5359 | 0.4580 | 14.5% better | EEM full + kNN |
| Glucose | 0.5589 | 0.5589 | 0.0% | EEM interpretable + kNN |

Filtered preprocessing/PARAFAC extension did not beat the filtered main-model baselines: rhamnose 0.4589 RMSE, xylose 0.5526 RMSE, glucose 0.7187 RMSE.

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
- `supervised_monosaccharides/comprehensive_modeling_report.html`: comprehensive visual report for processing, modelling, training, optimization, feature input/output tables, predicted-vs-true plots, residuals, and metrics.
- `supervised_monosaccharides/monosaccharide_softsensor_comprehensive_report.docx`: Word-format scientific report summarizing the current data, methods, modelling results, strengths, weaknesses, and biological interpretation.
- `supervised_monosaccharides_exclude_rha5/comprehensive_modeling_report.html`: HTML report for the filtered training/testing run with all `Rha (5)` examples excluded and comparison against the previous report.
- `supervised_monosaccharides_exclude_rha5/monosaccharide_softsensor_exclude_rha5_report.docx`: Word report for the filtered run.
- `supervised_monosaccharides/optimization_improvement_summary.csv`: final-best RMSE compared with the initial baseline search.
- `generate_supervised_visual_report.py`: pure-Python SVG/HTML report generator for supervised modelling visualizations.
- `preprocessing_raman.py`: pure-NumPy Raman preprocessing and feature export pipeline with explicit preprocessing configuration labels.
- `eem_parafac_features.py`: pure-NumPy EEM PARAFAC feature export and rank-selection workflow.
- `train_preprocessed_models.py`: compact model comparison for Raman-preprocessed, EEM-PARAFAC, and fused feature sets versus the last best models.
- `rhamnose_ml/scripts/train_baseline.py`: starter baseline training entry point for supervised Rhamnose prediction.
- `rhamnose_ml/src/rhamnose_ml/train.py`: baseline training pipeline using PLS with grouped train/test splitting.

## Current blockers / bugs
- No quantitative HPLC monosaccharide target file is currently available for culture-sample supervised training.
- `scikit-learn`, `pandas`, and `matplotlib` are not installed in the current environment, so advanced analysis was implemented with lower-level tooling.
- Some directory listings in PowerShell returned stale snapshots; direct file existence checks were used as fallback validation.

## Next recommended steps
1. Obtain and merge the actual quantitative HPLC monosaccharide reference table for culture samples.
2. Re-run supervised training with culture targets and compare against the current standards/spikes calibration results.
3. Refine the already-added Raman ALS baseline correction, EEM scatter-region masking, and PARAFAC score export using tested libraries if dependency installation is allowed.
4. If external dependencies are allowed, compare the pure-NumPy search against scikit-learn PLSR/SVR/XGBoost implementations. The current pure-NumPy optimization did not deliver an extra 20% RMSE reduction beyond the previous best.

## How to run
```bash
pip install -r rhamnose_ml/requirements.txt
python build_enriched_inventory.py
python visualize_eem_raman.py
python export_rhamnose_features.py
python explore_features_unsupervised.py
python train_monosaccharide_softsensor.py
python preprocessing_raman.py
python eem_parafac_features.py
python train_preprocessed_models.py
python generate_supervised_visual_report.py
python generate_docx_model_report.py
$env:EXCLUDE_RHA5='1'; $env:SUPERVISED_OUT_DIR='supervised_monosaccharides_exclude_rha5'; python train_monosaccharide_softsensor.py
$env:EXCLUDE_RHA5='1'; $env:SUPERVISED_OUT_DIR='supervised_monosaccharides_exclude_rha5'; python train_preprocessed_models.py
$env:EXCLUDE_RHA5='1'; $env:SUPERVISED_OUT_DIR='supervised_monosaccharides_exclude_rha5'; python generate_supervised_visual_report.py
$env:EXCLUDE_RHA5='1'; $env:SUPERVISED_OUT_DIR='supervised_monosaccharides_exclude_rha5'; python generate_docx_model_report.py
python rhamnose_ml/scripts/train_baseline.py --config rhamnose_ml/config/defaults.json
```
