import csv
import math
import os
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
FEATURES_DIR = ROOT / "features"
OUT_DIR = ROOT / os.environ.get("SUPERVISED_OUT_DIR", "supervised_monosaccharides")
INTERPRETABLE_CSV = FEATURES_DIR / "rhamnose_interpretable_features.csv"
FULL_CSV = FEATURES_DIR / "rhamnose_full_feature_matrix.csv"
EXCLUDE_RHA5 = os.environ.get("EXCLUDE_RHA5", "").strip().lower() in {"1", "true", "yes"}

TARGETS = ("rhamnose_gL", "xylose_gL", "glucose_gL")
PREVIOUS_BEST_RMSE = {
    "rhamnose_gL": 0.72189956,
    "xylose_gL": 0.53806598,
    "glucose_gL": 0.59378797,
}
ID_COLUMNS = (
    "sample_set",
    "batch",
    "replicate",
    "sample_id",
    "container_type",
    "metadata_experiment",
    "metadata_plate",
    "metadata_well",
    "legend_treatment_label",
)
MERGE_KEY_COLUMNS = (
    "batch",
    "replicate",
    "container_type",
    "metadata_experiment",
    "metadata_plate",
    "metadata_well",
    "legend_treatment_label",
)


def safe_float(value):
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_targets(label):
    text = (label or "").strip()
    if not text:
        return {}, "unlabelled"
    low = text.lower()
    if low in {"blank f/2", "milliq", "algae (blank)"}:
        return {target: 0.0 for target in TARGETS}, "blank_or_matrix_blank"

    match = re.search(r"\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)", text)
    if not match:
        return {}, "non_numeric_treatment"
    value = float(match.group(1))

    prefix = text[: match.start()].strip().lower().replace(" ", "")
    targets = {}

    if prefix == "rha":
        targets["rhamnose_gL"] = value
        targets["xylose_gL"] = 0.0
        targets["glucose_gL"] = 0.0
    elif prefix == "xyl":
        targets["rhamnose_gL"] = 0.0
        targets["xylose_gL"] = value
        targets["glucose_gL"] = 0.0
    elif prefix == "glu":
        targets["rhamnose_gL"] = 0.0
        targets["xylose_gL"] = 0.0
        targets["glucose_gL"] = value
    elif prefix in {"mmf/2", "mmalgae"}:
        targets["rhamnose_gL"] = value
        targets["xylose_gL"] = value
        targets["glucose_gL"] = value
    elif prefix == "rha-glu":
        targets["rhamnose_gL"] = value
        targets["xylose_gL"] = 0.0
        targets["glucose_gL"] = value
    elif prefix == "rha-xyl":
        targets["rhamnose_gL"] = value
        targets["xylose_gL"] = value
        targets["glucose_gL"] = 0.0
    elif prefix == "rha-algae":
        targets["rhamnose_gL"] = value
    else:
        return {}, "unsupported_numeric_treatment"
    return targets, "known_standard_or_spike"


def target_rows(rows):
    out = []
    for row in rows:
        targets, source = parse_targets(row.get("legend_treatment_label", ""))
        new = dict(row)
        new["target_source"] = source
        for target in TARGETS:
            new[target] = "" if target not in targets else f"{targets[target]:.8g}"
        out.append(new)
    return out


def is_excluded_rha5(row):
    if not EXCLUDE_RHA5:
        return False
    label = (row.get("legend_treatment_label") or "").strip().lower().replace(" ", "")
    return label == "rha(5)" and safe_float(row.get("rhamnose_gL")) == 5.0


def merge_modalities(rows):
    merged = {}
    order = []
    for row in rows:
        key = tuple(row.get(col, "") for col in MERGE_KEY_COLUMNS)
        if key not in merged:
            merged[key] = dict(row)
            order.append(key)
            continue
        current = merged[key]
        for col, value in row.items():
            text = "" if value is None else str(value)
            if text.strip() and not str(current.get(col, "")).strip():
                current[col] = value
        # Prefer positive modality flags and file paths from either row.
        for col in ("eem_available", "raman_available", "eem_present", "raman_present"):
            if str(row.get(col, "")).lower() == "true":
                current[col] = row.get(col, "")
        for col in ("eem_file", "raman_file"):
            if str(row.get(col, "")).strip():
                current[col] = row.get(col, "")
    return [merged[key] for key in order]


def feature_columns(fieldnames, feature_set):
    if feature_set == "raman_interpretable":
        return [
            c
            for c in fieldnames
            if c.startswith("raman_") or c.startswith("r735_") or c.startswith("r905_")
            or c.startswith("r1156_") or c.startswith("r1408_") or c.startswith("r1523_")
            or c.startswith("r1878_")
        ]
    if feature_set == "eem_interpretable":
        return [c for c in fieldnames if c.startswith("eem_")]
    if feature_set == "fusion_interpretable":
        return feature_columns(fieldnames, "raman_interpretable") + feature_columns(fieldnames, "eem_interpretable")
    if feature_set == "raman_full":
        return [c for c in fieldnames if c.startswith("raman_") and c[6:].replace("-", "").isdigit()]
    if feature_set == "eem_full":
        return [c for c in fieldnames if c.startswith("eem_ex")]
    if feature_set == "fusion_full":
        return feature_columns(fieldnames, "raman_full") + feature_columns(fieldnames, "eem_full")
    raise ValueError(f"Unknown feature set: {feature_set}")


def build_matrix(rows, columns, target):
    x_rows = []
    y_vals = []
    kept_rows = []
    for row in rows:
        y = safe_float(row.get(target))
        if not math.isfinite(y):
            continue
        values = [safe_float(row.get(col)) for col in columns]
        if not any(math.isfinite(v) for v in values):
            continue
        x_rows.append(values)
        y_vals.append(y)
        kept_rows.append(row)
    if not x_rows:
        return np.empty((0, len(columns))), np.empty((0,)), []
    return np.asarray(x_rows, dtype=float), np.asarray(y_vals, dtype=float), kept_rows


def is_target_relevant(row, target):
    label = (row.get("legend_treatment_label") or "").lower().replace(" ", "")
    if label in {"blankf/2", "milliq", "algae(blank)"}:
        return True
    if target == "rhamnose_gL":
        return any(token in label for token in ("rha", "mm"))
    if target == "xylose_gL":
        return any(token in label for token in ("xyl", "mm"))
    if target == "glucose_gL":
        return any(token in label for token in ("glu", "mm"))
    return True


def median_impute_fit(x):
    with np.errstate(all="ignore"):
        med = np.nanmedian(x, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return med


def apply_impute(x, med):
    out = x.copy()
    inds = np.where(~np.isfinite(out))
    out[inds] = np.take(med, inds[1])
    return out


def standardize_fit(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return mean, std


def standardize_apply(x, mean, std):
    return (x - mean) / std


def variance_filter_fit(x, min_std=1e-10):
    return x.std(axis=0) > min_std


def corr_select_fit(x, y, max_features):
    y_center = y - y.mean()
    x_center = x - x.mean(axis=0)
    denom = np.sqrt(np.sum(x_center * x_center, axis=0) * np.sum(y_center * y_center))
    corr = np.divide(np.abs(x_center.T @ y_center), denom, out=np.zeros(x.shape[1]), where=denom > 0)
    order = np.argsort(corr)[::-1]
    keep_n = min(max_features, x.shape[1])
    keep = np.zeros(x.shape[1], dtype=bool)
    keep[order[:keep_n]] = True
    return keep


def ridge_fit(x, y, alpha):
    xtx = x.T @ x
    reg = np.eye(x.shape[1]) * alpha
    coef = np.linalg.pinv(xtx + reg) @ x.T @ y
    intercept = y.mean() - x.mean(axis=0) @ coef
    return intercept, coef


def ridge_predict(model, x):
    intercept, coef = model
    return intercept + x @ coef


def pcr_fit(x, y, n_components, alpha):
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    n = min(n_components, vt.shape[0])
    components = vt[:n].T
    scores = x @ components
    ridge = ridge_fit(scores, y, alpha)
    return components, ridge


def pcr_predict(model, x):
    components, ridge = model
    return ridge_predict(ridge, x @ components)


def pls1_fit(x, y, n_components, alpha):
    x_res = x.copy()
    y_res = (y - y.mean()).reshape(-1, 1)
    weights = []
    loadings = []
    q_vals = []
    for _ in range(min(n_components, x.shape[1], x.shape[0] - 1)):
        w = x_res.T @ y_res
        norm = float(np.linalg.norm(w))
        if norm < 1e-12:
            break
        w = w / norm
        t = x_res @ w
        denom = float((t.T @ t).item())
        if denom < 1e-12:
            break
        p = (x_res.T @ t) / denom
        q = float(((y_res.T @ t) / denom).item())
        x_res = x_res - t @ p.T
        y_res = y_res - t * q
        weights.append(w.reshape(-1))
        loadings.append(p.reshape(-1))
        q_vals.append(q)
    if not weights:
        return None
    w_mat = np.column_stack(weights)
    p_mat = np.column_stack(loadings)
    q = np.asarray(q_vals)
    rotations = w_mat @ np.linalg.pinv(p_mat.T @ w_mat)
    scores = x @ rotations
    ridge = ridge_fit(scores, y, alpha)
    return rotations, ridge


def pls1_predict(model, x):
    rotations, ridge = model
    return ridge_predict(ridge, x @ rotations)


def pairwise_distances(x_train, row, metric):
    if metric == "manhattan":
        return np.sum(np.abs(x_train - row), axis=1)
    if metric == "cosine":
        train_norm = np.linalg.norm(x_train, axis=1)
        row_norm = float(np.linalg.norm(row))
        denom = np.maximum(train_norm * row_norm, 1e-12)
        sim = (x_train @ row) / denom
        return 1.0 - np.clip(sim, -1.0, 1.0)
    if metric == "correlation":
        train_center = x_train - x_train.mean(axis=1, keepdims=True)
        row_center = row - row.mean()
        train_norm = np.linalg.norm(train_center, axis=1)
        row_norm = float(np.linalg.norm(row_center))
        denom = np.maximum(train_norm * row_norm, 1e-12)
        sim = (train_center @ row_center) / denom
        return 1.0 - np.clip(sim, -1.0, 1.0)
    return np.sqrt(np.sum((x_train - row) ** 2, axis=1))


def knn_predict(x_train, y_train, x_test, k, weighted, metric):
    preds = []
    k = min(k, len(y_train))
    for row in x_test:
        dist = pairwise_distances(x_train, row, metric)
        order = np.argsort(dist)[:k]
        vals = y_train[order]
        if weighted:
            weights = 1.0 / np.maximum(dist[order], 1e-9)
            preds.append(float(np.sum(weights * vals) / np.sum(weights)))
        else:
            preds.append(float(np.mean(vals)))
    return np.asarray(preds, dtype=float)


def kernel_matrix(x_left, x_right, kernel, gamma):
    if kernel == "linear":
        return x_left @ x_right.T
    diff = x_left[:, None, :] - x_right[None, :, :]
    if kernel == "laplacian":
        dist = np.sum(np.abs(diff), axis=2)
        return np.exp(-gamma * dist)
    dist2 = np.sum(diff * diff, axis=2)
    return np.exp(-gamma * dist2)


def median_distance_gamma(x, kernel, gamma_scale):
    if len(x) < 2:
        return 1.0
    diffs = x[:, None, :] - x[None, :, :]
    if kernel == "laplacian":
        dist = np.sum(np.abs(diffs), axis=2)
    else:
        dist = np.sqrt(np.sum(diffs * diffs, axis=2))
    vals = dist[np.triu_indices_from(dist, k=1)]
    vals = vals[np.isfinite(vals) & (vals > 1e-12)]
    if len(vals) == 0:
        return 1.0
    median = float(np.median(vals))
    if kernel == "laplacian":
        return gamma_scale / max(median, 1e-12)
    return gamma_scale / max(median * median, 1e-12)


def kernel_ridge_fit(x_train, y_train, alpha, kernel, gamma_scale):
    gamma = median_distance_gamma(x_train, kernel, gamma_scale)
    k_train = kernel_matrix(x_train, x_train, kernel, gamma)
    dual = np.linalg.pinv(k_train + np.eye(len(x_train)) * alpha) @ y_train
    return x_train, dual, kernel, gamma


def kernel_ridge_predict(model, x_test):
    x_train, dual, kernel, gamma = model
    return kernel_matrix(x_test, x_train, kernel, gamma) @ dual


def transform_y(y, transform):
    if transform == "log1p":
        return np.log1p(y)
    return y


def transform_x(x, transform):
    if transform == "sample_center":
        return x - x.mean(axis=1, keepdims=True)
    if transform == "sample_zscore":
        centered = x - x.mean(axis=1, keepdims=True)
        scale = x.std(axis=1, keepdims=True)
        return centered / np.where(scale > 1e-12, scale, 1.0)
    if transform == "l2":
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(norm > 1e-12, norm, 1.0)
    return x


def inverse_transform_y(y, transform):
    if transform == "log1p":
        return np.maximum(np.expm1(y), 0.0)
    return y


def split_indices(rows, seed):
    groups = np.asarray([
        f"{r.get('metadata_plate')}-{r.get('metadata_well')}-{r.get('legend_treatment_label')}"
        for r in rows
    ])
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * 0.25)))
    test_groups = set(unique[:n_test])
    test = np.asarray([idx for idx, group in enumerate(groups) if group in test_groups], dtype=int)
    train = np.asarray([idx for idx, group in enumerate(groups) if group not in test_groups], dtype=int)
    if len(test) == 0 or len(train) < 5:
        idx = np.arange(len(rows))
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * 0.25)))
        return idx[n_test:], idx[:n_test]
    return train, test


def metrics(y_true, y_pred):
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))
    baseline = float(np.sqrt(np.mean((y_true - y_true.mean()) ** 2)))
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "test_mean_rmse": baseline}


def prepare_train_test(x, y, train_idx, test_idx, max_features, x_transform):
    x_train_raw = x[train_idx]
    x_test_raw = x[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    finite_keep = np.isfinite(x_train_raw).any(axis=0)
    if not finite_keep.any():
        return None
    x_train_raw = x_train_raw[:, finite_keep]
    x_test_raw = x_test_raw[:, finite_keep]
    med = median_impute_fit(x_train_raw)
    x_train = apply_impute(x_train_raw, med)
    x_test = apply_impute(x_test_raw, med)
    var_keep = variance_filter_fit(x_train)
    x_train = x_train[:, var_keep]
    x_test = x_test[:, var_keep]
    if x_train.shape[1] == 0:
        return None
    corr_keep = corr_select_fit(x_train, y_train, max_features=max_features)
    x_train = x_train[:, corr_keep]
    x_test = x_test[:, corr_keep]
    mean, std = standardize_fit(x_train)
    x_train = standardize_apply(x_train, mean, std)
    x_test = standardize_apply(x_test, mean, std)
    x_train = transform_x(x_train, x_transform)
    x_test = transform_x(x_test, x_transform)
    return x_train, x_test, y_train, y_test, int(var_keep.sum()), int(corr_keep.sum())


def evaluate_feature_set(rows, columns, target, feature_set, cohort):
    if cohort == "target_focused":
        rows = [row for row in rows if is_target_relevant(row, target)]
    x, y, kept = build_matrix(rows, columns, target)
    if len(kept) < 12:
        return [], []

    focused_nonlinear = (
        (target == "rhamnose_gL" and cohort in {"all_known", "target_focused"} and feature_set == "fusion_full")
        or (target == "xylose_gL" and cohort == "all_known" and feature_set == "eem_full")
        or (target == "glucose_gL" and cohort == "target_focused" and feature_set == "eem_interpretable")
        or feature_set.startswith("raman_preprocessed")
        or feature_set.startswith("eem_parafac")
        or feature_set.startswith("parafac_raman_fusion")
    )
    configs = []
    max_feature_options = [12, 24, 48, 96, 192, 384]
    for max_features in max_feature_options:
        if feature_set.endswith("interpretable") and max_features > 96:
            continue
        for y_transform in ("none", "log1p"):
            for x_transform in ("none",):
                configs.append(("ridge", {"alpha": 0.1, "max_features": max_features, "y_transform": y_transform, "x_transform": x_transform}))
                configs.append(("ridge", {"alpha": 1.0, "max_features": max_features, "y_transform": y_transform, "x_transform": x_transform}))
                configs.append(("ridge", {"alpha": 10.0, "max_features": max_features, "y_transform": y_transform, "x_transform": x_transform}))
            for k in (1, 3, 5, 9):
                distance_metrics = ("euclidean", "manhattan", "cosine", "correlation") if focused_nonlinear else ("euclidean",)
                x_transforms = ("none", "sample_center", "sample_zscore", "l2") if focused_nonlinear else ("none",)
                for metric in distance_metrics:
                    for x_transform in x_transforms:
                        configs.append(("knn", {"k": k, "weighted": True, "metric": metric, "max_features": max_features, "y_transform": y_transform, "x_transform": x_transform}))
            if focused_nonlinear:
                for kernel in ("rbf", "laplacian"):
                    for alpha in (0.01, 0.1, 1.0):
                        for gamma_scale in (0.5, 1.0, 2.0):
                            for x_transform in ("none", "l2"):
                                configs.append(("krr", {"kernel": kernel, "alpha": alpha, "gamma_scale": gamma_scale, "max_features": max_features, "y_transform": y_transform, "x_transform": x_transform}))
            for x_transform in ("none",):
                for n in (2, 3, 5, 8, 12):
                    configs.append(("pcr", {"n_components": n, "alpha": 1.0, "max_features": max_features, "y_transform": y_transform, "x_transform": x_transform}))
                    configs.append(("pls", {"n_components": n, "alpha": 1.0, "max_features": max_features, "y_transform": y_transform, "x_transform": x_transform}))

    metric_rows = []
    prediction_rows = []
    for seed in range(10):
        train_idx, test_idx = split_indices(kept, seed=1000 + seed)
        for model_name, params in configs:
            prepared = prepare_train_test(
                x,
                y,
                train_idx,
                test_idx,
                max_features=params["max_features"],
                x_transform=params.get("x_transform", "none"),
            )
            if prepared is None:
                continue
            x_train, x_test, y_train, y_test, n_var, n_selected = prepared
            if x_train.shape[0] <= 3 or x_test.shape[0] <= 0:
                continue
            y_transform = params.get("y_transform", "none")
            y_train_fit = transform_y(y_train, y_transform)
            try:
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
                        continue
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
                    continue
            except np.linalg.LinAlgError:
                continue

            m = metrics(y_test, pred)
            config_label = ",".join(f"{k}={v}" for k, v in params.items())
            metric_rows.append(
                {
                    "target": target,
                    "cohort": cohort,
                    "feature_set": feature_set,
                    "model": model_name,
                    "config": config_label,
                    "seed": seed,
                    "n_rows": len(kept),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_features_raw": len(columns),
                    "n_features_after_variance": n_var,
                    "n_features_selected": n_selected,
                    "rmse": m["rmse"],
                    "mae": m["mae"],
                    "r2": m["r2"],
                    "test_mean_rmse": m["test_mean_rmse"],
                    "improvement_vs_test_mean_pct": 100.0 * (m["test_mean_rmse"] - m["rmse"]) / m["test_mean_rmse"]
                    if m["test_mean_rmse"] > 0
                    else float("nan"),
                }
            )

            # Keep the prediction artifact compact; full model comparisons are in metrics CSVs.
            if seed == 0 and focused_nonlinear and model_name in {"knn", "krr"}:
                for local_idx, row_idx in enumerate(test_idx):
                    src = kept[row_idx]
                    pred_row = {key: src.get(key, "") for key in ID_COLUMNS}
                    pred_row.update(
                        {
                            "target": target,
                            "cohort": cohort,
                            "feature_set": feature_set,
                            "model": model_name,
                            "config": config_label,
                            "y_true": f"{float(y_test[local_idx]):.8g}",
                            "y_pred": f"{float(pred[local_idx]):.8g}",
                            "residual": f"{float(y_test[local_idx] - pred[local_idx]):.8g}",
                        }
                    )
                    prediction_rows.append(pred_row)
    return metric_rows, prediction_rows


def aggregate_metrics(metric_rows):
    buckets = {}
    for row in metric_rows:
        key = (row["target"], row["cohort"], row["feature_set"], row["model"], row["config"])
        buckets.setdefault(key, []).append(row)
    out = []
    for key, rows in buckets.items():
        target, cohort, feature_set, model, config = key
        rmse = np.asarray([float(r["rmse"]) for r in rows], dtype=float)
        mae = np.asarray([float(r["mae"]) for r in rows], dtype=float)
        r2 = np.asarray([float(r["r2"]) for r in rows], dtype=float)
        imp = np.asarray([float(r["improvement_vs_test_mean_pct"]) for r in rows], dtype=float)
        out.append(
            {
                "target": target,
                "cohort": cohort,
                "feature_set": feature_set,
                "model": model,
                "config": config,
                "n_repeats": len(rows),
                "mean_rmse": float(np.nanmean(rmse)),
                "std_rmse": float(np.nanstd(rmse)),
                "mean_mae": float(np.nanmean(mae)),
                "mean_r2": float(np.nanmean(r2)),
                "mean_improvement_vs_test_mean_pct": float(np.nanmean(imp)),
            }
        )
    return sorted(out, key=lambda r: (r["target"], r["mean_rmse"]))


def best_by_target(aggregated):
    out = {}
    for row in aggregated:
        out.setdefault(row["target"], row)
    return out


def optimization_summary(aggregated):
    rows = []
    for target in TARGETS:
        baseline_candidates = [
            row
            for row in aggregated
            if row["target"] == target
            and row["cohort"] == "all_known"
            and row["model"] in {"ridge", "pcr", "pls"}
            and "y_transform=none" in row["config"]
        ]
        final_candidates = [row for row in aggregated if row["target"] == target]
        if not baseline_candidates or not final_candidates:
            continue
        baseline = min(baseline_candidates, key=lambda row: row["mean_rmse"])
        final = min(final_candidates, key=lambda row: row["mean_rmse"])
        improvement = 100.0 * (baseline["mean_rmse"] - final["mean_rmse"]) / baseline["mean_rmse"]
        previous_best = PREVIOUS_BEST_RMSE.get(target, float("nan"))
        additional_improvement = (
            100.0 * (previous_best - final["mean_rmse"]) / previous_best
            if previous_best > 0
            else float("nan")
        )
        rows.append(
            {
                "target": target,
                "previous_best_rmse": previous_best,
                "initial_baseline_feature_set": baseline["feature_set"],
                "initial_baseline_model": baseline["model"],
                "initial_baseline_config": baseline["config"],
                "initial_baseline_rmse": baseline["mean_rmse"],
                "final_best_cohort": final["cohort"],
                "final_best_feature_set": final["feature_set"],
                "final_best_model": final["model"],
                "final_best_config": final["config"],
                "final_best_rmse": final["mean_rmse"],
                "rmse_improvement_pct": improvement,
                "additional_rmse_improvement_vs_previous_best_pct": additional_improvement,
                "met_additional_20pct_threshold": additional_improvement >= 20.0,
                "met_10pct_threshold": improvement >= 10.0,
            }
        )
    return rows


def make_html_report(target_summary, aggregated, best, opt_rows):
    top_rows = sorted(aggregated, key=lambda r: r["mean_rmse"])[:40]
    best_cards = "\n".join(
        f"""
        <section class="card">
          <h3>{target}</h3>
          <p><strong>{row['cohort']} / {row['feature_set']} + {row['model']}</strong></p>
          <p>{row['config']}</p>
          <p>Mean RMSE: {row['mean_rmse']:.4g}; MAE: {row['mean_mae']:.4g}; R2: {row['mean_r2']:.3f}</p>
          <p>Mean improvement vs test-mean predictor: {row['mean_improvement_vs_test_mean_pct']:.1f}%</p>
        </section>
        """
        for target, row in best.items()
    )
    table_rows = "\n".join(
        f"<tr><td>{r['target']}</td><td>{r['cohort']}</td><td>{r['feature_set']}</td><td>{r['model']}</td><td>{r['config']}</td>"
        f"<td>{r['mean_rmse']:.5g}</td><td>{r['mean_mae']:.5g}</td><td>{r['mean_r2']:.3f}</td>"
        f"<td>{r['mean_improvement_vs_test_mean_pct']:.1f}%</td></tr>"
        for r in top_rows
    )
    summary_rows = "\n".join(
        f"<tr><td>{r['target']}</td><td>{r['labelled_rows']}</td><td>{r['known_nonzero_rows']}</td>"
        f"<td>{r['min_value']}</td><td>{r['max_value']}</td></tr>"
        for r in target_summary
    )
    opt_table_rows = "\n".join(
        f"<tr><td>{r['target']}</td><td>{r['initial_baseline_rmse']:.5g}</td><td>{r['final_best_rmse']:.5g}</td>"
        f"<td>{r['rmse_improvement_pct']:.1f}%</td><td>{'yes' if r['met_10pct_threshold'] else 'no'}</td>"
        f"<td>{r['additional_rmse_improvement_vs_previous_best_pct']:.1f}%</td>"
        f"<td>{'yes' if r['met_additional_20pct_threshold'] else 'no'}</td>"
        f"<td>{r['final_best_cohort']} / {r['final_best_feature_set']} / {r['final_best_model']}</td></tr>"
        for r in opt_rows
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Supervised Monosaccharide Soft Sensor Results</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 0; color: #17202a; background: #f6f8fb; line-height: 1.5; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 22px; }}
    h1 {{ margin-bottom: 6px; }}
    .muted {{ color: #5d6d7e; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    .card {{ background: #fff; border: 1px solid #d6dde5; border-radius: 8px; padding: 14px; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 14px; }}
    th, td {{ border: 1px solid #d6dde5; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #eef3f8; }}
    svg {{ max-width: 100%; height: auto; background: #fff; border: 1px solid #d6dde5; border-radius: 8px; }}
    @media (max-width: 860px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>Supervised Monosaccharide Soft Sensor Results</h1>
  <p class="muted">Standards and known spikes were parsed from treatment labels in the enriched inventory. Culture rows without quantitative HPLC targets remain excluded from supervised fitting. {'The Rha (5) examples were excluded from this run.' if EXCLUDE_RHA5 else ''}</p>

  <h2>Best Models Found</h2>
  <div class="grid">{best_cards}</div>

  <h2>Optimization Improvement</h2>
  <table>
    <tr><th>Target</th><th>Initial baseline RMSE</th><th>Final best RMSE</th><th>RMSE improvement</th><th>Met 10% threshold</th><th>Additional improvement vs previous best</th><th>Met extra 20%</th><th>Final model</th></tr>
    {opt_table_rows}
  </table>

  <h2>Labelled Target Coverage</h2>
  <table>
    <tr><th>Target</th><th>Labelled rows</th><th>Non-zero rows</th><th>Minimum g/L</th><th>Maximum g/L</th></tr>
    {summary_rows}
  </table>

  <h2>Training Flow</h2>
  <svg viewBox="0 0 1100 340" role="img" aria-label="Supervised training flow">
    <defs><marker id="a" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="#44546a"/></marker></defs>
    <rect x="25" y="55" width="180" height="80" rx="8" fill="#e7f4f4" stroke="#2f7d7e"/><text x="115" y="88" text-anchor="middle" font-size="16" font-weight="700">EEM features</text><text x="115" y="114" text-anchor="middle" font-size="13">hotspots or unfolded matrix</text>
    <rect x="25" y="205" width="180" height="80" rx="8" fill="#f0e9f7" stroke="#7a4e9f"/><text x="115" y="238" text-anchor="middle" font-size="16" font-weight="700">Raman features</text><text x="115" y="264" text-anchor="middle" font-size="13">windows or full spectrum</text>
    <rect x="285" y="130" width="190" height="90" rx="8" fill="#fff" stroke="#b8c2cc"/><text x="380" y="163" text-anchor="middle" font-size="16" font-weight="700">Target parser</text><text x="380" y="190" text-anchor="middle" font-size="13">Rha / Xyl / Glu g/L</text>
    <rect x="555" y="130" width="190" height="90" rx="8" fill="#eef5ff" stroke="#2d5f9a"/><text x="650" y="163" text-anchor="middle" font-size="16" font-weight="700">Model search</text><text x="650" y="190" text-anchor="middle" font-size="13">Ridge, PCR, PLS</text>
    <rect x="825" y="130" width="230" height="90" rx="8" fill="#fff8ef" stroke="#b5651d"/><text x="940" y="163" text-anchor="middle" font-size="16" font-weight="700">Predicted concentrations</text><text x="940" y="190" text-anchor="middle" font-size="13">Rhamnose, xylose, glucose</text>
    <line x1="205" y1="95" x2="285" y2="155" stroke="#44546a" stroke-width="2" marker-end="url(#a)"/>
    <line x1="205" y1="245" x2="285" y2="195" stroke="#44546a" stroke-width="2" marker-end="url(#a)"/>
    <line x1="475" y1="175" x2="555" y2="175" stroke="#44546a" stroke-width="2" marker-end="url(#a)"/>
    <line x1="745" y1="175" x2="825" y2="175" stroke="#44546a" stroke-width="2" marker-end="url(#a)"/>
  </svg>

  <h2>Top Model Configurations</h2>
  <table>
    <tr><th>Target</th><th>Cohort</th><th>Feature set</th><th>Model</th><th>Config</th><th>Mean RMSE</th><th>Mean MAE</th><th>Mean R2</th><th>Improvement</th></tr>
    {table_rows}
  </table>
</main>
</body>
</html>
"""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    interp_rows = merge_modalities(target_rows(read_csv(INTERPRETABLE_CSV)))
    full_rows = merge_modalities(target_rows(read_csv(FULL_CSV)))
    if EXCLUDE_RHA5:
        interp_rows = [row for row in interp_rows if not is_excluded_rha5(row)]
        full_rows = [row for row in full_rows if not is_excluded_rha5(row)]

    labelled_fields = list(interp_rows[0].keys())
    labelled_target_path = (
        OUT_DIR / "monosaccharide_interpretable_targets_exclude_rha5.csv"
        if EXCLUDE_RHA5
        else FEATURES_DIR / "monosaccharide_interpretable_targets.csv"
    )
    write_csv(labelled_target_path, interp_rows, labelled_fields)

    target_summary = []
    for target in TARGETS:
        vals = np.asarray([safe_float(row.get(target)) for row in interp_rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        target_summary.append(
            {
                "target": target,
                "labelled_rows": len(vals),
                "known_nonzero_rows": int(np.sum(vals > 0)),
                "min_value": "" if len(vals) == 0 else f"{float(np.min(vals)):.8g}",
                "max_value": "" if len(vals) == 0 else f"{float(np.max(vals)):.8g}",
            }
        )

    all_metrics = []
    all_predictions = []
    feature_plan = [
        ("raman_interpretable", interp_rows, list(interp_rows[0].keys())),
        ("eem_interpretable", interp_rows, list(interp_rows[0].keys())),
        ("fusion_interpretable", interp_rows, list(interp_rows[0].keys())),
        ("raman_full", full_rows, list(full_rows[0].keys())),
        ("eem_full", full_rows, list(full_rows[0].keys())),
        ("fusion_full", full_rows, list(full_rows[0].keys())),
    ]
    for feature_set, rows, fields in feature_plan:
        cols = feature_columns(fields, feature_set)
        if not cols:
            continue
        for target in TARGETS:
            for cohort in ("all_known", "target_focused"):
                metrics_rows, prediction_rows = evaluate_feature_set(rows, cols, target, feature_set, cohort)
                all_metrics.extend(metrics_rows)
                all_predictions.extend(prediction_rows)

    if not all_metrics:
        raise RuntimeError("No supervised metrics were generated. Check target parsing and feature columns.")

    metric_fields = list(all_metrics[0].keys())
    write_csv(OUT_DIR / "model_search_metrics_by_split.csv", all_metrics, metric_fields)

    aggregated = aggregate_metrics(all_metrics)
    agg_fields = list(aggregated[0].keys())
    write_csv(
        OUT_DIR / "model_search_metrics_summary.csv",
        [
            {key: (f"{value:.8g}" if isinstance(value, float) else value) for key, value in row.items()}
            for row in aggregated
        ],
        agg_fields,
    )

    best = best_by_target(aggregated)
    opt_rows = optimization_summary(aggregated)
    write_csv(
        OUT_DIR / "best_models.csv",
        [
            {key: (f"{value:.8g}" if isinstance(value, float) else value) for key, value in row.items()}
            for row in best.values()
        ],
        agg_fields,
    )

    write_csv(
        OUT_DIR / "optimization_improvement_summary.csv",
        [
            {key: (f"{value:.8g}" if isinstance(value, float) else value) for key, value in row.items()}
            for row in opt_rows
        ],
        list(opt_rows[0].keys()),
    )

    if all_predictions:
        pred_fields = list(all_predictions[0].keys())
        write_csv(OUT_DIR / "example_predictions_seed0.csv", all_predictions, pred_fields)

    write_csv(OUT_DIR / "target_summary.csv", target_summary, list(target_summary[0].keys()))
    (OUT_DIR / "supervised_report.html").write_text(
        make_html_report(target_summary, aggregated, best, opt_rows),
        encoding="utf-8",
    )

    print(f"Generated {len(all_metrics)} model/split metrics")
    for target, row in best.items():
        print(
            f"{target}: {row['feature_set']} {row['model']} {row['config']} "
            f"RMSE={row['mean_rmse']:.5g} improvement={row['mean_improvement_vs_test_mean_pct']:.1f}%"
        )
    for row in opt_rows:
        print(
            f"{row['target']} optimized RMSE improvement vs initial baseline: "
            f"{row['rmse_improvement_pct']:.1f}% "
            f"(additional vs previous best: {row['additional_rmse_improvement_vs_previous_best_pct']:.1f}%)"
        )


if __name__ == "__main__":
    main()
