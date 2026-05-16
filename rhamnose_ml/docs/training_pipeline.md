# Training Pipeline

## High-Level Flow

```mermaid
flowchart TD
    A[Enriched inventory CSV] --> B[Filter rows with numeric Rhamnose target]
    C[Raw EEM CSVs] --> D[EEM parser and flattening]
    E[Raw Raman CSVs] --> F[Raman parser and 500-2000 cm-1 crop]
    B --> G[Sample alignment by row]
    D --> G
    F --> G
    G --> H[Train/test split by batch or experiment]
    H --> I1[EEM-only PLS]
    H --> I2[Raman-only PLS]
    H --> I3[Fused EEM+Raman PLS]
    I1 --> J[Metrics and predictions]
    I2 --> J
    I3 --> J
```

## Input-Output Explanation

### Inputs per sample

- `EEM`: excitation-emission intensity matrix
- `Raman`: intensity spectrum from `500-2000 cm^-1`
- metadata: experiment, plate, well, batch
- target: numeric HPLC Rhamnose concentration

### Outputs per sample

- predicted Rhamnose concentration
- residual = actual - predicted
- model provenance: `eem`, `raman`, or `fusion`

## Feature Construction

### EEM

```text
raw CSV
  -> parse matrix
  -> convert OVER to NaN
  -> keep intensity block only
  -> flatten to 1D feature vector
```

### Raman

```text
raw CSV
  -> parse shift and intensity columns
  -> crop to 500-2000 cm^-1
  -> keep intensity values on the observed grid
```

## Baseline Model

The initial baseline is **Partial Least Squares Regression**:

- good for high-dimensional spectroscopy data
- robust to multicollinearity
- interpretable compared with deeper models

## Recommended Progression

1. Establish the PLS baseline.
2. Compare `EEM`, `Raman`, and `Fusion`.
3. Add preprocessing variants.
4. Only then test tree models or neural multimodal models.
