import csv
import importlib.util
import math
import os
from pathlib import Path

import numpy as np

from train_monosaccharide_softsensor import (
    FEATURES_DIR,
    FULL_CSV,
    ID_COLUMNS,
    INTERPRETABLE_CSV,
    TARGETS,
    aggregate_metrics,
    apply_impute,
    build_matrix,
    feature_columns,
    median_impute_fit,
    merge_modalities,
    metrics,
    safe_float,
    split_indices,
    standardize_apply,
    standardize_fit,
    target_rows,
    transform_y,
    inverse_transform_y,
    variance_filter_fit,
    write_csv,
)


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / os.environ.get("SUPERVISED_OUT_DIR", "supervised_monosaccharides")
OUT_CSV = OUT_DIR / "dependency_model_comparison.csv"
OUT_SUMMARY = OUT_DIR / "dependency_model_comparison_summary.csv"
OUT_PREDICTIONS = OUT_DIR / "dependency_model_predictions_seed0.csv"
OUT_HTML = OUT_DIR / "dependency_model_comparison.html"
EXCLUDE_RHA5 = os.environ.get("EXCLUDE_RHA5", "").strip().lower() in {"1", "true", "yes"}


def dependency_status():
    return {
        "scipy": bool(importlib.util.find_spec("scipy")),
        "sklearn": bool(importlib.util.find_spec("sklearn")),
        "xgboost": bool(importlib.util.find_spec("xgboost")),
        "tensorly": bool(importlib.util.find_spec("tensorly")),
    }


def select_features(x_train, x_test, max_features):
    med = median_impute_fit(x_train)
    x_train = apply_impute(x_train, med)
    x_test = apply_impute(x_test, med)
    keep = variance_filter_fit(x_train)
    x_train = x_train[:, keep]
    x_test = x_test[:, keep]
    if x_train.shape[1] == 0:
        return None
    y_proxy = np.arange(x_train.shape[0], dtype=float)
    # Dependency models perform their own regularization, so use variance-ranked columns
    # here rather than target-ranked leakage-prone selection across packages.
    var_order = np.argsort(x_train.var(axis=0))[::-1][: min(max_features, x_train.shape[1])]
    x_train = x_train[:, var_order]
    x_test = x_test[:, var_order]
    mean, std = standardize_fit(x_train)
    return standardize_apply(x_train, mean, std), standardize_apply(x_test, mean, std)


def evaluate_sklearn_model(model_name, model, x_train, y_train, x_test):
    if model_name == "sklearn_plsr":
        return model.fit(x_train, y_train).predict(x_test).reshape(-1)
    return model.fit(x_train, y_train).predict(x_test)


def run_dependency_models():
    status = dependency_status()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not status["sklearn"]:
        rows = [
            {
                "dependency": dep,
                "available": str(available),
                "action": "Install dependency to enable comparison" if not available else "Available",
            }
            for dep, available in status.items()
        ]
        write_csv(OUT_CSV, rows, ["dependency", "available", "action"])
        OUT_HTML.write_text(
            "<!doctype html><html><body><h1>Dependency Model Comparison</h1>"
            "<p>scikit-learn is not installed, so PLSR/SVR comparisons could not be run. "
            "XGBoost also requires its package to be installed. The pure-NumPy reports remain the active results.</p>"
            + "<table><tr><th>Dependency</th><th>Available</th><th>Action</th></tr>"
            + "".join(f"<tr><td>{r['dependency']}</td><td>{r['available']}</td><td>{r['action']}</td></tr>" for r in rows)
            + "</table></body></html>",
            encoding="utf-8",
        )
        print("scikit-learn is not installed; wrote dependency availability report.")
        return

    from sklearn.cross_decomposition import PLSRegression
    from sklearn.svm import SVR

    xgb_available = status["xgboost"]
    if xgb_available:
        from xgboost import XGBRegressor
    else:
        XGBRegressor = None

    interp_rows = merge_modalities(target_rows(read_csv(INTERPRETABLE_CSV)))
    full_rows = merge_modalities(target_rows(read_csv(FULL_CSV)))
    if EXCLUDE_RHA5:
        interp_rows = [row for row in interp_rows if not is_excluded_rha5(row)]
        full_rows = [row for row in full_rows if not is_excluded_rha5(row)]
    plans = [
        ("eem_interpretable", interp_rows, list(interp_rows[0].keys())),
        ("fusion_interpretable", interp_rows, list(interp_rows[0].keys())),
        ("eem_full", full_rows, list(full_rows[0].keys())),
        ("fusion_full", full_rows, list(full_rows[0].keys())),
    ]
    all_metrics = []
    all_predictions = []
    for feature_set, rows, fields in plans:
        cols = feature_columns(fields, feature_set)
        if not cols:
            continue
        for target in TARGETS:
            x, y, kept = build_matrix(rows, cols, target)
            if len(kept) < 12:
                continue
            for seed in range(5):
                train_idx, test_idx = split_indices(kept, seed=2000 + seed)
                if len(train_idx) < 8 or len(test_idx) < 2:
                    continue
                prepared = select_features(x[train_idx].copy(), x[test_idx].copy(), 192)
                if prepared is None:
                    continue
                x_train, x_test = prepared
                y_train, y_test = y[train_idx], y[test_idx]
                models = [
                    ("sklearn_plsr", "n_components=2,max_features=192", PLSRegression(n_components=min(2, x_train.shape[1]))),
                    ("sklearn_plsr", "n_components=5,max_features=192", PLSRegression(n_components=min(5, x_train.shape[1]))),
                    ("sklearn_svr", "kernel=rbf,C=10,epsilon=0.05,max_features=192", SVR(kernel="rbf", C=10.0, epsilon=0.05)),
                    ("sklearn_svr", "kernel=linear,C=1,epsilon=0.05,max_features=192", SVR(kernel="linear", C=1.0, epsilon=0.05)),
                ]
                if XGBRegressor is not None:
                    models.append(
                        (
                            "xgboost",
                            "n_estimators=200,max_depth=2,learning_rate=0.05,max_features=192",
                            XGBRegressor(
                                n_estimators=200,
                                max_depth=2,
                                learning_rate=0.05,
                                subsample=0.9,
                                colsample_bytree=0.9,
                                objective="reg:squarederror",
                                random_state=seed,
                                n_jobs=1,
                            ),
                        )
                    )
                for model_name, config, model in models:
                    try:
                        pred = evaluate_sklearn_model(model_name, model, x_train, y_train, x_test)
                    except Exception:
                        continue
                    m = metrics(y_test, np.asarray(pred, dtype=float))
                    all_metrics.append(
                        {
                            "target": target,
                            "cohort": "all_known",
                            "feature_set": feature_set,
                            "model": model_name,
                            "config": config,
                            "seed": seed,
                            "rmse": m["rmse"],
                            "mae": m["mae"],
                            "r2": m["r2"],
                            "improvement_vs_test_mean_pct": 100.0 * (m["test_mean_rmse"] - m["rmse"]) / m["test_mean_rmse"]
                            if m["test_mean_rmse"] > 0
                            else float("nan"),
                        }
                    )
                    if seed == 0:
                        pred_arr = np.asarray(pred, dtype=float)
                        for local_idx, row_idx in enumerate(test_idx):
                            src = kept[row_idx]
                            pred_row = {key: src.get(key, "") for key in ID_COLUMNS}
                            pred_row.update(
                                {
                                    "target": target,
                                    "cohort": "all_known",
                                    "feature_set": feature_set,
                                    "model": model_name,
                                    "config": config,
                                    "y_true": f"{float(y_test[local_idx]):.8g}",
                                    "y_pred": f"{float(pred_arr[local_idx]):.8g}",
                                    "residual": f"{float(y_test[local_idx] - pred_arr[local_idx]):.8g}",
                                }
                            )
                            all_predictions.append(pred_row)
    if not all_metrics:
        raise RuntimeError("Dependency models were available but produced no metrics.")
    summary = aggregate_metrics(all_metrics)
    write_csv(OUT_CSV, all_metrics, list(all_metrics[0].keys()))
    write_csv(
        OUT_SUMMARY,
        [{key: (f"{value:.8g}" if isinstance(value, float) else value) for key, value in row.items()} for row in summary],
        list(summary[0].keys()),
    )
    if all_predictions:
        write_csv(OUT_PREDICTIONS, all_predictions, list(all_predictions[0].keys()))
    rows = "".join(
        f"<tr><td>{r['target']}</td><td>{r['feature_set']}</td><td>{r['model']}</td><td>{r['config']}</td><td>{r['mean_rmse']:.4f}</td><td>{r['mean_r2']:.3f}</td></tr>"
        for r in sorted(summary, key=lambda row: row["mean_rmse"])[:30]
    )
    OUT_HTML.write_text(
        "<!doctype html><html><body><h1>Dependency Model Comparison</h1>"
        "<table><tr><th>Target</th><th>Feature set</th><th>Model</th><th>Config</th><th>RMSE</th><th>R2</th></tr>"
        + rows
        + "</table></body></html>",
        encoding="utf-8",
    )
    print(f"Wrote {OUT_SUMMARY}")


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def is_excluded_rha5(row):
    label = (row.get("legend_treatment_label") or "").strip().lower().replace(" ", "")
    return label == "rha(5)" and safe_float(row.get("rhamnose_gL")) == 5.0


if __name__ == "__main__":
    run_dependency_models()
