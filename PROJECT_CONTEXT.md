# PROJECT_CONTEXT.md

## Purpose
Build a scientifically defensible soft sensor for predicting monosaccharide concentrations from Emilie SoftSensor spectroscopy data.

Primary target variables:
- `rhamnose_gL`
- `xylose_gL`
- `glucose_gL`

Primary input modalities:
- EEM fluorescence matrices
- Raman spectra
- Metadata and plate legends for experiment, plate, well, sample type, and known standard/spike concentrations

## Data Layout
Raw data were copied into the repository under:

`data/raw/Emilie_SoftSensor/`

Important source descriptions:
- `data/raw/Emilie_SoftSensor/Data_descriptions/Data description MS SoftSensor.docx`
- `data/raw/Emilie_SoftSensor/Data_descriptions/Experiment legend.xlsx`
- `data/raw/Emilie_SoftSensor/Data_descriptions/Emilie_HPLC_Sample legend.xlsx`
- `data/raw/Emilie_SoftSensor/Data_descriptions/metadata_Emilie_SoftSensor_Rhamnose_*.xlsx`

Important generated project files:
- `eem_raman_hplc_inventory_enriched.csv`
- `features/rhamnose_interpretable_features.csv`
- `features/rhamnose_full_feature_matrix.csv`
- `features/monosaccharide_interpretable_targets.csv`
- `supervised_monosaccharides/supervised_report.html`
- `supervised_monosaccharides/optimization_improvement_summary.csv`

## Spectroscopy Facts
EEM:
- One CSV file per sample.
- Matrix is 15 rows x 19 columns.
- Row headers are excitation wavelengths in nm.
- Column headers are emission wavelengths in nm.
- Values are fluorescence intensity.
- `OVER` means detector saturation beyond the linear range and should be masked or treated as unreliable.

Raman:
- One CSV file per sample.
- Two columns with no header: Raman shift in cm-1 and Raman intensity.
- Files span 99-3500 cm-1.
- Useful measured range is described as 500-2000 cm-1.
- Excitation wavelength was 532 nm.

Standards:
- `Rha` means rhamnose.
- `Xyl` means xylose.
- `Glu` means glucose.
- `MM` means rhamnose, xylose, and glucose mixed at the bracketed final concentration.
- Bracketed concentrations are in g/L.

## Current Modelling Status
Supervised modelling currently uses known standards and spikes parsed from treatment labels.

Best RMSE improvements versus the initial linear/no-log/all-known baseline:
- Rhamnose: 16.4% improvement, best model `fusion_full + weighted kNN`.
- Xylose: 11.7% improvement, best model `eem_full + weighted kNN + log target`.
- Glucose: 22.7% improvement, best model `eem_interpretable + weighted kNN`.

Culture-sample supervised prediction is not complete because quantitative HPLC monosaccharide target values are not yet available in the project.

## Repository
Remote:

`https://github.com/nhaduy99/Soft_sesor_glucose.git`

Branch:

`main`
