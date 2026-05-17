# SESSION_HANDOFF.md

## Current State
The project is on GitHub and synced on branch `main`.

Latest committed baseline before this handoff:

`4ec8ec1 Add Raman preprocessing and EEM PARAFAC models`

Current supervised calibration outputs are in:

`supervised_monosaccharides/`

Open the main supervised report here:

`supervised_monosaccharides/supervised_report.html`

Open the comprehensive visual modelling report here:

`supervised_monosaccharides/comprehensive_modeling_report.html`

Open the Word-format scientific report here:

`supervised_monosaccharides/monosaccharide_softsensor_comprehensive_report.docx`

Filtered `Rha (5)` exclusion outputs are in:

`supervised_monosaccharides_exclude_rha5/`

Open the filtered HTML report here:

`supervised_monosaccharides_exclude_rha5/comprehensive_modeling_report.html`

Open the filtered Word report here:

`supervised_monosaccharides_exclude_rha5/monosaccharide_softsensor_exclude_rha5_report.docx`

Open the filtered dependency-aware Word report here:

`supervised_monosaccharides_exclude_rha5/monosaccharide_softsensor_exclude_rha5_refined_dependencies_report.docx`

Open the refined dependency-aware Word report here:

`supervised_monosaccharides/monosaccharide_softsensor_refined_dependencies_report.docx`

Open the dependency availability/comparison report here:

`supervised_monosaccharides/dependency_model_comparison.html`

## What Was Completed
- Uploaded the project to GitHub.
- Uploaded the raw data folder into `data/raw/Emilie_SoftSensor`.
- Read the `Data_descriptions` folder and created `docs/softsensor_monosaccharide_model_io.html`.
- Built a pure-NumPy supervised training script: `train_monosaccharide_softsensor.py`.
- Parsed known standard/spike concentrations into:
  - `rhamnose_gL`
  - `xylose_gL`
  - `glucose_gL`
- Ran iterative model search over EEM, Raman, and fusion features.
- Met the requested minimum 10% RMSE improvement threshold versus the initial baseline:
  - Rhamnose: 16.4%
  - Xylose: 11.7%
  - Glucose: 22.7%
- Added project maintenance files: `PROJECT_CONTEXT.md`, `DECISIONS.md`, and `SESSION_HANDOFF.md`.
- Added an explicit rule to `AGENTS.md`: at the end of every task, update `SESSION_HANDOFF.md` so the next Codex session can continue without chat history.
- Continued optimization with distance-normalized kNN and focused RBF/Laplacian kernel ridge. Latest confirmed results did not reach the requested extra 20% RMSE reduction beyond the previous best.
- Generated `supervised_monosaccharides/comprehensive_modeling_report.html` with predicted-vs-true scatter plots, residual plots, metric bar charts, top-model tables, and pipeline documentation.
- Inserted processed spectroscopy visual samples into `supervised_monosaccharides/comprehensive_modeling_report.html`: annotated Raman overlays and EEM heatmaps with analysis of monosaccharide signal interpretation.
- Added `preprocessing_raman.py` for Raman cosmic-spike removal, ALS baseline correction, Savitzky-Golay smoothing/derivatives, SNV, optional area normalization, and `preprocessing_config` tracking.
- Added `eem_parafac_features.py` for cleaned EEM cube PARAFAC rank 2-8 fitting, rank selection, score/loadings export, and component SVG plots.
- Ran `preprocessing_raman.py`, `eem_parafac_features.py`, and `train_preprocessed_models.py`.
- Latest new-feature comparison: glucose improved 11.6% against the last best model using `raman_preprocessed_als_sg2_snv`; rhamnose and xylose did not improve.
- Updated `supervised_monosaccharides/comprehensive_modeling_report.html` with Raman preprocessing/PARAFAC result tables and PARAFAC loading/component-map plots.
- Added an explicit feature input/output table to `supervised_monosaccharides/comprehensive_modeling_report.html`, covering Raman interpretable/full/preprocessed features, EEM interpretable/unfolded/PARAFAC features, fusion inputs, and predicted concentration outputs.
- Extended `train_preprocessed_models.py` with a focused kernel-ridge pass over the strongest preprocessed Raman and Raman+PARAFAC feature sets.
- Regenerated `supervised_monosaccharides/preprocessed_model_best_vs_last.csv`, `supervised_monosaccharides/preprocessed_model_search_metrics_summary.csv`, `supervised_monosaccharides/preprocessed_model_search_metrics_by_split.csv`, and the comprehensive HTML report.
- Latest focused preprocessed/PARAFAC result: glucose improved to 0.5094 RMSE, an 8.8% gain versus the latest project-level baseline. Rhamnose and xylose did not meet the requested 5% improvement threshold.
- Added `generate_docx_model_report.py` and generated `supervised_monosaccharides/monosaccharide_softsensor_comprehensive_report.docx`. The report is structured as a 6-8 page Word document with pipeline visualisation, RMSE/improvement plots, predicted-vs-true plots, PARAFAC visuals, strengths, weaknesses, biological interpretation, and recommended next steps.
- Added `EXCLUDE_RHA5` and `SUPERVISED_OUT_DIR` modes to the training/report scripts, reran training/testing after excluding all `Rha (5)` examples, and generated new filtered HTML/DOCX reports under `supervised_monosaccharides_exclude_rha5/`.
- Refined preprocessing/model scripts for stronger dependencies when available:
  - `preprocessing_raman.py` now uses SciPy sparse ALS and SciPy Savitzky-Golay if SciPy is installed, with NumPy fallback.
  - `eem_parafac_features.py` now records PARAFAC backend, masks primary and second-order scatter regions, and uses TensorLy non-negative PARAFAC if TensorLy is installed, with NumPy CP-ALS fallback.
# Session Handoff - 2026-05-17 Report Visualization Update

## Latest Completed Task
Updated the filtered reports for the `Rha (5)` exclusion workflow with clearer beginner explanations, detailed pipeline diagrams, and Nature-style plot refinements.

Files changed in this task:
- `generate_supervised_visual_report.py`
- `generate_docx_model_report.py`
- `eem_parafac_features.py`
- `features/eem_parafac_exclude_rha5/rank6_excitation_loadings.svg`
- `features/eem_parafac_exclude_rha5/rank6_emission_loadings.svg`
- `features/eem_parafac_exclude_rha5/rank6_component*_map.svg`
- `supervised_monosaccharides_exclude_rha5/comprehensive_modeling_report.html`
- `supervised_monosaccharides_exclude_rha5/monosaccharide_softsensor_exclude_rha5_refined_dependencies_report.docx`
- `PROGRESS.md`
- `TODO.md`
- `SESSION_HANDOFF.md`

What changed:
- Added plain-language explanation of the SciPy Raman preprocessing path: cosmic-spike removal, ALS baseline correction, Savitzky-Golay smoothing/derivatives, SNV, and optional area normalization.
- Added plain-language explanation of EEM PARAFAC: EEM cube decomposition into sample scores, excitation loadings, and emission loadings.
- Added a detailed end-to-end pipeline diagram for EEM/Raman inputs, preprocessing, feature extraction, fusion, model comparison, and monosaccharide predictions.
- Restyled report SVG plots with clearer x/y axes, light gridlines, target-specific colors, and figure captions.
- Regenerated filtered PARAFAC rank-6 loading plots and component maps with axis labels and explanatory notes.

Commands run:
```powershell
conda run -n base python -m py_compile eem_parafac_features.py generate_supervised_visual_report.py generate_docx_model_report.py
$env:EXCLUDE_RHA5='1'; conda run -n base python eem_parafac_features.py
$env:EXCLUDE_RHA5='1'; $env:SUPERVISED_OUT_DIR='supervised_monosaccharides_exclude_rha5'; conda run -n base python train_preprocessed_models.py
$env:EXCLUDE_RHA5='1'; $env:SUPERVISED_OUT_DIR='supervised_monosaccharides_exclude_rha5'; conda run -n base python compare_dependency_models.py
$env:EXCLUDE_RHA5='1'; $env:SUPERVISED_OUT_DIR='supervised_monosaccharides_exclude_rha5'; conda run -n base python generate_supervised_visual_report.py
$env:EXCLUDE_RHA5='1'; $env:SUPERVISED_OUT_DIR='supervised_monosaccharides_exclude_rha5'; $env:REPORT_DOCX_NAME='monosaccharide_softsensor_exclude_rha5_refined_dependencies_report.docx'; conda run -n base python generate_docx_model_report.py
conda run -n base python -c "from html.parser import HTMLParser; from pathlib import Path; p=Path('supervised_monosaccharides_exclude_rha5/comprehensive_modeling_report.html'); HTMLParser().feed(p.read_text(encoding='utf-8')); print('html ok', p.stat().st_size)"
conda run -n base python -c "import zipfile, xml.etree.ElementTree as ET; p='supervised_monosaccharides_exclude_rha5/monosaccharide_softsensor_exclude_rha5_refined_dependencies_report.docx'; z=zipfile.ZipFile(p); ET.fromstring(z.read('word/document.xml')); print('docx ok', len([n for n in z.namelist() if n.startswith('word/media/')]))"
```

Validation:
- HTML parsed successfully: `html ok 343381`.
- DOCX parsed successfully: `docx ok 12`.
- Filtered PARAFAC still selected rank 6 with TensorLy.
- Filtered preprocessed/PARAFAC comparison remains worse than filtered main baselines: rhamnose 0.4933 RMSE, xylose 0.5544 RMSE, glucose 0.7058 RMSE.
- Dependency-backed filtered comparison still writes `supervised_monosaccharides_exclude_rha5/dependency_model_comparison_summary.csv`.

Exact next command to continue later:
```powershell
git status --short
```

Recommended next work:
- Inspect the refreshed HTML/DOCX visually in a browser/Word.
- Commit and push the current report/figure updates.
- Continue the scientific interpretation only after a quantitative HPLC culture-sample target table is available.

---

  - `compare_dependency_models.py` now runs scikit-learn PLSR/SVR and XGBoost comparisons when those packages are installed.
- Current dependency check: SciPy, scikit-learn, XGBoost, TensorLy, pandas, and matplotlib are not installed, so dependency-backed model comparison could not run. `supervised_monosaccharides/dependency_model_comparison.csv` records this.
- Regenerated Raman preprocessing, EEM PARAFAC, focused preprocessed/PARAFAC results, and the comprehensive HTML report. The original DOCX could not be overwritten because Windows denied access, so the refreshed Word report was saved as `supervised_monosaccharides/monosaccharide_softsensor_refined_dependencies_report.docx`.
- Modified the latest dependency-aware workflow for the `Rha (5)` exclusion case:
  - `eem_parafac_features.py` now supports `EXCLUDE_RHA5=1` and writes filtered PARAFAC outputs to `features/eem_parafac_exclude_rha5/` plus `features/eem_parafac_scores_exclude_rha5.csv`.
  - `train_preprocessed_models.py`, `generate_supervised_visual_report.py`, `generate_docx_model_report.py`, and `compare_dependency_models.py` use the filtered PARAFAC artifacts when `EXCLUDE_RHA5=1`.
  - Regenerated filtered PARAFAC, filtered preprocessed/PARAFAC metrics, filtered dependency availability report, filtered HTML report, and `supervised_monosaccharides_exclude_rha5/monosaccharide_softsensor_exclude_rha5_refined_dependencies_report.docx`.
- User instructed to use Conda base for all Python runs. Verified base Python is `C:\ProgramData\anaconda3\python.exe`.
- Installed XGBoost and TensorLy into the Conda base Python user site using `conda run -n base python -m pip install --user xgboost tensorly`, because direct `conda install -n base ...` failed with `EnvironmentNotWritableError` for `C:\ProgramData\anaconda3`.
- Verified Conda base imports: SciPy, scikit-learn, XGBoost, TensorLy, pandas, matplotlib all import successfully.
- Re-ran filtered Raman preprocessing with SciPy path, filtered PARAFAC with TensorLy, filtered preprocessed/PARAFAC comparison, dependency-backed scikit-learn/XGBoost comparison, and regenerated filtered HTML/DOCX reports.

## Latest Confirmed Results
| Target | Best cohort | Best feature set | Best model | RMSE | Improvement vs initial baseline | Extra improvement vs previous best |
|---|---|---|---|---:|---:|---:|
| Rhamnose | target_focused | fusion_full | kNN, Manhattan, L2 row normalization | 0.7043 | 18.4% | 2.4% |
| Xylose | all_known | eem_full | Laplacian kernel ridge, log target | 0.5359 | 12.0% | 0.4% |
| Glucose | target_focused | eem_interpretable | kNN, Manhattan | 0.5589 | 27.3% | 5.9% |

The interrupted mean-blend optimization run was stopped and should not be treated as a confirmed result.

## Latest Preprocessed/PARAFAC Extension Results
| Target | Best new feature set | Model | RMSE | Latest project baseline RMSE | Improvement vs latest baseline | Met 5% |
|---|---|---|---:|---:|---:|---|
| Rhamnose | parafac_raman_fusion_als_sg0_snv_area | weighted kNN | 0.8001 | 0.7043 | -13.6% | No |
| Xylose | raman_preprocessed_als_sg2_snv | weighted kNN | 0.6050 | 0.5359 | -12.9% | No |
| Glucose | raman_preprocessed_als_sg2_snv | Laplacian kernel ridge | 0.5094 | 0.5589 | 8.8% | Yes |

## Latest `Rha (5)` Exclusion Results
The filtered target table `supervised_monosaccharides_exclude_rha5/monosaccharide_interpretable_targets_exclude_rha5.csv` has zero rows with `rhamnose_gL = 5`.

| Target | Previous best RMSE | Filtered best RMSE | Change vs previous | Filtered best model |
|---|---:|---:|---:|---|
| Rhamnose | 0.7043 | 0.4136 | 41.3% better | EEM full + ridge |
| Xylose | 0.5359 | 0.4580 | 14.5% better | EEM full + kNN |
| Glucose | 0.5589 | 0.5589 | 0.0% | EEM interpretable + kNN |

Filtered preprocessing/PARAFAC comparison did not beat the filtered main-model baselines: rhamnose 0.4589 RMSE, xylose 0.5526 RMSE, glucose 0.7187 RMSE.

Filtered dependency-aware PARAFAC after excluding `Rha (5)` now selected rank 6 with backend `tensorly_non_negative_parafac` and scatter mask `primary_nm=20.0;second_order_nm=25.0`. Latest filtered refined preprocessing/PARAFAC results:

| Target | Best filtered refined feature set | Model | RMSE | Compared with filtered main baseline |
|---|---|---|---:|---:|
| Rhamnose | PARAFAC + Raman fusion, ALS SG2 SNV | weighted kNN | 0.4933 | worse than 0.4136 |
| Xylose | preprocessed Raman, ALS SG2 SNV | weighted kNN | 0.5544 | worse than 0.4580 |
| Glucose | PARAFAC + Raman fusion, ALS SG0 SNV area | RBF KRR | 0.7058 | worse than 0.5589 |

Dependency-backed filtered comparison now runs. Best visible result:

| Target | Feature set | Model | RMSE | Note |
|---|---|---|---:|---|
| Rhamnose | fusion_full | scikit-learn PLSR, 5 components | 0.3598 | better than filtered pure-NumPy rhamnose best 0.4136 |

No quantitative HPLC concentration target table was found beyond the existing sample legends/generated target files, so culture-target merging remains blocked.

## Latest Dependency-Aware Refinement Results
Current dependency availability:
- SciPy: unavailable
- scikit-learn: unavailable
- XGBoost: unavailable
- TensorLy: unavailable

Refined EEM PARAFAC used `numpy_cp_als`, applied `primary_nm=20.0;second_order_nm=25.0`, and selected rank 2. Focused preprocessing/PARAFAC comparison after refinement:

| Target | Best refined feature set | Model | RMSE | Change vs latest project baseline |
|---|---|---|---:|---:|
| Rhamnose | PARAFAC + Raman fusion, ALS SG0 SNV area | weighted kNN | 0.8070 | -14.6% |
| Xylose | preprocessed Raman, ALS SG2 SNV | weighted kNN | 0.6050 | -12.9% |
| Glucose | preprocessed Raman, ALS SG2 SNV | Laplacian KRR | 0.5094 | +8.8% |

## Commands Recently Run
```bash
python train_monosaccharide_softsensor.py
python -m py_compile train_monosaccharide_softsensor.py
python -c "from html.parser import HTMLParser; HTMLParser().feed(open('supervised_monosaccharides/supervised_report.html', encoding='utf-8').read()); print('HTML parse ok')"
git push
python train_monosaccharide_softsensor.py
python preprocessing_raman.py
python eem_parafac_features.py
python train_preprocessed_models.py
python generate_supervised_visual_report.py
python -m py_compile train_preprocessed_models.py generate_supervised_visual_report.py preprocessing_raman.py eem_parafac_features.py
python -c "from html.parser import HTMLParser; from pathlib import Path; p=Path('supervised_monosaccharides/comprehensive_modeling_report.html'); HTMLParser().feed(p.read_text(encoding='utf-8')); print('html ok', p.stat().st_size)"
python generate_docx_model_report.py
python -c "import zipfile, xml.etree.ElementTree as ET; p='supervised_monosaccharides/monosaccharide_softsensor_comprehensive_report.docx'; z=zipfile.ZipFile(p); ET.fromstring(z.read('word/document.xml')); print('docx ok', len([n for n in z.namelist() if n.startswith('word/media/')]))"
$env:EXCLUDE_RHA5='1'; $env:SUPERVISED_OUT_DIR='supervised_monosaccharides_exclude_rha5'; python train_monosaccharide_softsensor.py
$env:EXCLUDE_RHA5='1'; $env:SUPERVISED_OUT_DIR='supervised_monosaccharides_exclude_rha5'; python train_preprocessed_models.py
$env:EXCLUDE_RHA5='1'; $env:SUPERVISED_OUT_DIR='supervised_monosaccharides_exclude_rha5'; python generate_supervised_visual_report.py
$env:EXCLUDE_RHA5='1'; $env:SUPERVISED_OUT_DIR='supervised_monosaccharides_exclude_rha5'; python generate_docx_model_report.py
python -c "from html.parser import HTMLParser; from pathlib import Path; p=Path('supervised_monosaccharides_exclude_rha5/comprehensive_modeling_report.html'); HTMLParser().feed(p.read_text(encoding='utf-8')); print('html ok', p.stat().st_size)"
python -c "import zipfile, xml.etree.ElementTree as ET; p='supervised_monosaccharides_exclude_rha5/monosaccharide_softsensor_exclude_rha5_report.docx'; z=zipfile.ZipFile(p); ET.fromstring(z.read('word/document.xml')); print('docx ok', len([n for n in z.namelist() if n.startswith('word/media/')]))"
python preprocessing_raman.py
python eem_parafac_features.py
python train_preprocessed_models.py
python compare_dependency_models.py
$env:REPORT_DOCX_NAME='monosaccharide_softsensor_refined_dependencies_report.docx'; python generate_docx_model_report.py
$env:EXCLUDE_RHA5='1'; conda run -n base python eem_parafac_features.py
$env:EXCLUDE_RHA5='1'; $env:SUPERVISED_OUT_DIR='supervised_monosaccharides_exclude_rha5'; conda run -n base python train_preprocessed_models.py
$env:EXCLUDE_RHA5='1'; $env:SUPERVISED_OUT_DIR='supervised_monosaccharides_exclude_rha5'; conda run -n base python compare_dependency_models.py
$env:EXCLUDE_RHA5='1'; $env:SUPERVISED_OUT_DIR='supervised_monosaccharides_exclude_rha5'; conda run -n base python generate_supervised_visual_report.py
$env:EXCLUDE_RHA5='1'; $env:SUPERVISED_OUT_DIR='supervised_monosaccharides_exclude_rha5'; $env:REPORT_DOCX_NAME='monosaccharide_softsensor_exclude_rha5_refined_dependencies_report.docx'; conda run -n base python generate_docx_model_report.py
```

## Key Caveat
The current supervised results are based on standards and known spikes parsed from treatment labels. They are not final culture-sample prediction results.

The next major blocker is obtaining and merging quantitative HPLC monosaccharide concentrations for culture samples.

## Next Recommended Action
Add the quantitative HPLC target table, then rerun:

```bash
python build_enriched_inventory.py
python export_rhamnose_features.py
python train_monosaccharide_softsensor.py
```

After rerunning, compare culture-sample grouped validation against the current standards/spikes calibration result.

Also remember: `AGENTS.md` requires updating this `SESSION_HANDOFF.md` at the end of every task so the next Codex session can continue without chat history.
