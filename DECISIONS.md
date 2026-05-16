# DECISIONS.md

## Modelling Decisions

### Use standards and known spikes as supervised calibration targets
Decision: parse numeric targets from `legend_treatment_label` when labels contain known concentrations such as `Rha (5)`, `Xyl (0.1)`, `Glu (1)`, `MM f/2 (0.01)`, `Rha-Glu (0.1)`, and `Rha-Xyl (1)`.

Reason: the quantitative HPLC culture-sample reference table is not yet present, but standards and known spikes provide valid supervised calibration labels in g/L.

Consequence: current supervised metrics are calibration/spike metrics, not final culture-sample prediction metrics.

### Treat culture rows without HPLC targets as excluded from supervised training
Decision: do not invent culture targets for labels such as `N0`, `S1`, `B2`, or `STD`.

Reason: these labels identify sample/treatment groups, not measured monosaccharide concentrations.

Consequence: culture-sample soft-sensor performance remains blocked until quantitative HPLC targets are merged.

### Use pure NumPy training for current supervised search
Decision: implement Ridge, PCR, PLS1, and weighted kNN in `train_monosaccharide_softsensor.py`.

Reason: the current environment has `numpy` but does not have `scikit-learn`, `pandas`, or `matplotlib`.

Consequence: results are reproducible without installing dependencies, but should later be compared with scikit-learn PLSR/SVR/XGBoost if dependency installation is allowed.

### Merge Raman and EEM by sample identity for fusion
Decision: merge rows using experiment, plate, well, replicate, container type, and treatment label rather than sample-set name.

Reason: standards can be labelled `MSStandards` for Raman and `Pavlovagyrans` for EEM while representing the same experiment/plate/well/treatment.

Consequence: fusion rows now correctly pair Raman and EEM features for the same well.

### Use RMSE improvement versus initial baseline as stopping criterion
Decision: compare final best models against the initial linear/no-log/all-known baseline and stop when each target improves by at least 10% RMSE.

Result:
- Rhamnose improved 16.4%.
- Xylose improved 11.7%.
- Glucose improved 22.7%.

## Data Decisions

### Commit raw data directly to Git
Decision: upload `data/raw/Emilie_SoftSensor` to GitHub.

Reason: user explicitly requested uploading the data folder to the repo; no individual file exceeded GitHub's 100 MB limit.

Consequence: the repo contains approximately 5,866 raw-data files and is larger than a code-only repo.

### Keep generated reports and metrics in the repo
Decision: commit HTML reports, metrics CSVs, and feature CSVs.

Reason: this project is research-oriented and the user requested maintained progress. Generated outputs document progress and allow review without rerunning long scripts.

Consequence: future commits should avoid accidental duplication of large generated files unless outputs changed meaningfully.
