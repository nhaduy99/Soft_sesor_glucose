import os
from pathlib import Path

import numpy as np

from train_monosaccharide_softsensor import (
    FEATURES_DIR,
    FULL_CSV,
    ID_COLUMNS,
    INTERPRETABLE_CSV,
    OUT_DIR,
    build_matrix,
    feature_columns,
    inverse_transform_y,
    is_excluded_rha5,
    is_target_relevant,
    kernel_ridge_fit,
    kernel_ridge_predict,
    knn_predict,
    merge_modalities,
    pcr_fit,
    pcr_predict,
    pls1_fit,
    pls1_predict,
    prepare_train_test,
    read_csv,
    ridge_fit,
    ridge_predict,
    split_indices,
    target_rows,
    transform_y,
    write_csv,
)


OUT_PREDICTIONS = OUT_DIR / "best_model_predictions_seed0.csv"


def parse_config(config):
    out = {}
    for item in str(config).split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        value = value.strip()
        if value.lower() in {"true", "false"}:
            out[key.strip()] = value.lower() == "true"
            continue
        try:
            number = float(value)
            out[key.strip()] = int(number) if number.is_integer() else number
        except ValueError:
            out[key.strip()] = value
    return out


def feature_plan():
    interp_rows = merge_modalities(target_rows(read_csv(INTERPRETABLE_CSV)))
    full_rows = merge_modalities(target_rows(read_csv(FULL_CSV)))
    if os.environ.get("EXCLUDE_RHA5", "").strip().lower() in {"1", "true", "yes"}:
        interp_rows = [row for row in interp_rows if not is_excluded_rha5(row)]
        full_rows = [row for row in full_rows if not is_excluded_rha5(row)]
    return {
        "raman_interpretable": (interp_rows, list(interp_rows[0].keys())),
        "eem_interpretable": (interp_rows, list(interp_rows[0].keys())),
        "fusion_interpretable": (interp_rows, list(interp_rows[0].keys())),
        "raman_full": (full_rows, list(full_rows[0].keys())),
        "eem_full": (full_rows, list(full_rows[0].keys())),
        "fusion_full": (full_rows, list(full_rows[0].keys())),
    }


def predict_best_row(best, rows, fields):
    target = best["target"]
    if best["cohort"] == "target_focused":
        rows = [row for row in rows if is_target_relevant(row, target)]
    cols = feature_columns(fields, best["feature_set"])
    x, y, kept = build_matrix(rows, cols, target)
    train_idx, test_idx = split_indices(kept, seed=1000)
    params = parse_config(best["config"])
    prepared = prepare_train_test(
        x,
        y,
        train_idx,
        test_idx,
        max_features=params.get("max_features", 96),
        x_transform=params.get("x_transform", "none"),
    )
    if prepared is None:
        return []
    x_train, x_test, y_train, y_test, _, _ = prepared
    y_transform = params.get("y_transform", "none")
    y_train_fit = transform_y(y_train, y_transform)
    model_name = best["model"]
    if model_name == "ridge":
        model = ridge_fit(x_train, y_train_fit, alpha=params["alpha"])
        pred = inverse_transform_y(ridge_predict(model, x_test), y_transform)
    elif model_name == "pcr":
        model = pcr_fit(
            x_train,
            y_train_fit,
            n_components=min(params["n_components"], x_train.shape[0] - 1, x_train.shape[1]),
            alpha=params["alpha"],
        )
        pred = inverse_transform_y(pcr_predict(model, x_test), y_transform)
    elif model_name == "pls":
        model = pls1_fit(
            x_train,
            y_train_fit,
            n_components=min(params["n_components"], x_train.shape[0] - 1, x_train.shape[1]),
            alpha=params["alpha"],
        )
        if model is None:
            return []
        pred = inverse_transform_y(pls1_predict(model, x_test), y_transform)
    elif model_name == "knn":
        pred = inverse_transform_y(
            knn_predict(
                x_train,
                y_train_fit,
                x_test,
                k=params["k"],
                weighted=params["weighted"],
                metric=params["metric"],
            ),
            y_transform,
        )
    elif model_name == "krr":
        model = kernel_ridge_fit(
            x_train,
            y_train_fit,
            alpha=params["alpha"],
            kernel=params["kernel"],
            gamma_scale=params["gamma_scale"],
        )
        pred = inverse_transform_y(kernel_ridge_predict(model, x_test), y_transform)
    else:
        return []

    out = []
    pred = np.asarray(pred, dtype=float)
    for local_idx, row_idx in enumerate(test_idx):
        src = kept[row_idx]
        row = {key: src.get(key, "") for key in ID_COLUMNS}
        row.update(
            {
                "target": target,
                "cohort": best["cohort"],
                "feature_set": best["feature_set"],
                "model": model_name,
                "config": best["config"],
                "y_true": f"{float(y_test[local_idx]):.8g}",
                "y_pred": f"{float(pred[local_idx]):.8g}",
                "residual": f"{float(y_test[local_idx] - pred[local_idx]):.8g}",
            }
        )
        out.append(row)
    return out


def main():
    plans = feature_plan()
    rows = []
    for best in read_csv(OUT_DIR / "best_models.csv"):
        if best["feature_set"] not in plans:
            continue
        plan_rows, fields = plans[best["feature_set"]]
        rows.extend(predict_best_row(best, plan_rows, fields))
    if not rows:
        raise RuntimeError("No best-model predictions were generated.")
    write_csv(OUT_PREDICTIONS, rows, list(rows[0].keys()))
    print(f"Wrote {OUT_PREDICTIONS}")


if __name__ == "__main__":
    main()
