# Rhamnose-Only Model Comparison

This folder compares rhamnose prediction strategies using EEM, Raman, processed Raman, EEM PARAFAC, and fused inputs.

Current best result:

- Input strategy: `EEM full + PARAFAC`
- Model: `Random forest`
- Config: `n=250,max_depth=4`
- Mean grouped CV RMSE: `0.3438` g/L
- Mean grouped CV R2: `0.193`

Main files:

- `rhamnose_model_metrics_by_split.csv`
- `rhamnose_model_metrics_summary.csv`
- `rhamnose_model_predictions_split0.csv`
- `rhamnose_model_comparison_report.html`
- `rhamnose_model_comparison_report.docx`
- `figures/`

Run command:

```powershell
conda run -n base python rhamnose_only_model_comparison.py
```
