import numpy as np
import pandas as pd

from .io import parse_eem_csv, parse_raman_csv


def _safe_numeric_target(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def select_training_rows(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    out = df.copy()
    out[target_column] = _safe_numeric_target(out[target_column])
    out = out.loc[out[target_column].notna()].reset_index(drop=True)
    return out


def build_feature_matrices(df: pd.DataFrame, config: dict) -> dict[str, np.ndarray]:
    eem_vectors = []
    raman_vectors = []

    for _, row in df.iterrows():
        eem_file = row.get("eem_file")
        raman_file = row.get("raman_file")

        if config["eem"]["enabled"]:
            if isinstance(eem_file, str) and eem_file.strip():
                eem_matrix = parse_eem_csv(
                    eem_file,
                    replace_over_with_nan=config["eem"]["replace_over_with_nan"],
                )
                eem_vectors.append(eem_matrix.reshape(-1))
            else:
                eem_vectors.append(None)

        if config["raman"]["enabled"]:
            if isinstance(raman_file, str) and raman_file.strip():
                raman_vector = parse_raman_csv(
                    raman_file,
                    min_shift=float(config["raman"]["min_shift"]),
                    max_shift=float(config["raman"]["max_shift"]),
                )
                raman_vectors.append(raman_vector)
            else:
                raman_vectors.append(None)

    features = {}

    if config["eem"]["enabled"]:
        valid_len = next((len(v) for v in eem_vectors if v is not None), None)
        features["eem"] = np.vstack(
            [v if v is not None else np.full(valid_len, np.nan) for v in eem_vectors]
        )

    if config["raman"]["enabled"]:
        valid_len = next((len(v) for v in raman_vectors if v is not None), None)
        features["raman"] = np.vstack(
            [v if v is not None else np.full(valid_len, np.nan) for v in raman_vectors]
        )

    if "eem" in features and "raman" in features:
        features["fusion"] = np.hstack([features["eem"], features["raman"]])

    return features
