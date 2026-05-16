from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_feature_matrices, select_training_rows
from .io import ensure_output_dir, load_inventory


def _build_pipeline(n_components: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pls", PLSRegression(n_components=n_components)),
        ]
    )


def _split_indices(df: pd.DataFrame, config: dict) -> tuple[np.ndarray, np.ndarray]:
    split_cfg = config["split"]
    group_column = split_cfg.get("group_column")
    test_size = float(split_cfg.get("test_size", 0.2))
    random_state = int(split_cfg.get("random_state", 42))

    if group_column and group_column in df.columns:
        groups = df[group_column].fillna("missing_group").astype(str)
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(df, groups=groups))
        return train_idx, test_idx

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    return np.asarray(train_idx), np.asarray(test_idx)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def run_training(config: dict) -> None:
    df = load_inventory(config["inventory_csv"])
    target_column = config["target_column"]

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in inventory. "
            "Add the quantitative HPLC Rhamnose target first."
        )

    train_df = select_training_rows(df, target_column)
    if train_df.empty:
        raise ValueError(
            f"No numeric rows found in target column '{target_column}'. "
            "Populate the enriched inventory with quantitative Rhamnose values first."
        )

    feature_mats = build_feature_matrices(train_df, config)
    y = train_df[target_column].to_numpy(dtype=float)
    train_idx, test_idx = _split_indices(train_df, config)

    out_dir = ensure_output_dir(config["output_dir"])
    metrics_rows = []
    prediction_frames = []

    for feature_name, X in feature_mats.items():
        if X is None:
            continue

        n_components = int(config["models"][feature_name]["n_components"])
        pipeline = _build_pipeline(n_components=n_components)
        pipeline.fit(X[train_idx], y[train_idx])
        preds = pipeline.predict(X[test_idx]).reshape(-1)
        metrics = _evaluate(y[test_idx], preds)

        metrics_rows.append(
            {
                "model": feature_name,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
            }
        )

        pred_df = train_df.iloc[test_idx][config["id_columns"]].copy()
        pred_df["model"] = feature_name
        pred_df["y_true"] = y[test_idx]
        pred_df["y_pred"] = preds
        pred_df["residual"] = pred_df["y_true"] - pred_df["y_pred"]
        prediction_frames.append(pred_df)

        model_path = Path(out_dir) / f"{feature_name}_pls.joblib"
        joblib.dump(pipeline, model_path)

    if not metrics_rows:
        raise ValueError("No feature matrices were created. Check EEM/Raman file paths and config flags.")

    pd.DataFrame(metrics_rows).to_csv(Path(out_dir) / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        Path(out_dir) / "predictions.csv",
        index=False,
    )
