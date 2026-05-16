import csv
from pathlib import Path

import numpy as np
import pandas as pd


def load_inventory(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def parse_eem_csv(path: str, replace_over_with_nan: bool = True) -> np.ndarray:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            rows.append(row)

    data_rows = rows[1:]
    values = []
    for row in data_rows:
        numeric_cells = row[1:]
        parsed = []
        for cell in numeric_cells:
            if cell == "OVER" and replace_over_with_nan:
                parsed.append(np.nan)
            else:
                parsed.append(float(cell))
        values.append(parsed)
    return np.asarray(values, dtype=float)


def parse_raman_csv(path: str, min_shift: float, max_shift: float) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", dtype=float)
    shifts = arr[:, 0]
    intensities = arr[:, 1]
    mask = (shifts >= min_shift) & (shifts <= max_shift)
    return intensities[mask]


def ensure_output_dir(path: str) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
