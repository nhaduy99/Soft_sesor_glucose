# Raman and EEM Visual Summary

- Raman files analysed: 839
- EEM files analysed: 1330
- Source folders: `Raw_data_Plate` and `Raw_data_Flask` only; `Raw_data_files_ALL` was excluded to avoid duplication.

## Raman dominant peaks in the mean spectrum

- `735 cm^-1`: dominant mean-spectrum maximum, intensity `11`
- `905 cm^-1`: dominant mean-spectrum maximum, intensity `60`
- `1156 cm^-1`: dominant mean-spectrum maximum, intensity `205`
- `1408 cm^-1`: dominant mean-spectrum maximum, intensity `8`
- `1523 cm^-1`: dominant mean-spectrum maximum, intensity `296`
- `1878 cm^-1`: dominant mean-spectrum maximum, intensity `8`

## EEM dominant hotspots in the mean matrix

- `Ex 300 nm / Em 300 nm`: mean intensity `6194805`
- `Ex 360 nm / Em 340 nm`: mean intensity `4540170`
- `Ex 480 nm / Em 500 nm`: mean intensity `54140`
- `Ex 500 nm / Em 460 nm`: mean intensity `7290`
- `Ex 500 nm / Em 300 nm`: mean intensity `418`
- `Ex 440 nm / Em 640 nm`: mean intensity `399`

## Interpretation notes

- Raman annotations mark the strongest local maxima in the dataset-average spectrum. They are useful candidate regions for feature engineering and peak-ratio analysis.
- EEM hotspots identify excitation-emission coordinates with consistently strong fluorescence signal across the dataset.
- The maximum EEM detector saturation fraction observed locally is `1.00`; saturated regions should be masked or handled explicitly in downstream modeling.