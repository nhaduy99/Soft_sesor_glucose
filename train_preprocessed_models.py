import csv
import math
from pathlib import Path

import numpy as np

from train_monosaccharide_softsensor import (
    MERGE_KEY_COLUMNS,
    OUT_DIR,
    PREVIOUS_BEST_RMSE,
    TARGETS,
    is_target_relevant,
    pairwise_distances,
    safe_float,
    write_csv,
)


ROOT = Path(__file__).resolve().parent
RAMAN_CSV = ROOT / "features" / "raman_preprocessed_features.csv"
PARAFAC_CSV = ROOT / "features" / "eem_parafac_scores.csv"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def key_for(row):
    return tuple(row.get(col, "") for col in MERGE_KEY_COLUMNS)


def feature_cols(rows, prefixes):
    fields = rows[0].keys() if rows else []
    return [col for col in fields if any(col.startswith(prefix) for prefix in prefixes)]


def merge_feature_rows(left_rows, right_rows):
    right_by_key = {key_for(row): row for row in right_rows}
    out = []
    for left in left_rows:
        right = right_by_key.get(key_for(left))
        if not right:
            continue
        merged = dict(left)
        for key, value in right.items():
            if key.startswith("parafac_") or key in {"parafac_selected_rank"}:
                merged[key] = value
        out.append(merged)
    return out


def matrix(rows, cols, target, cohort):
    if cohort == "target_focused":
        rows = [row for row in rows if is_target_relevant(row, target)]
    kept, x_rows, y_vals = [], [], []
    for row in rows:
        y = safe_float(row.get(target))
        if not math.isfinite(y):
            continue
        vals = [safe_float(row.get(col)) for col in cols]
        if not any(math.isfinite(v) for v in vals):
            continue
        kept.append(row)
        x_rows.append(vals)
        y_vals.append(y)
    if not kept:
        return np.empty((0, len(cols))), np.empty((0,)), []
    return np.asarray(x_rows, dtype=float), np.asarray(y_vals, dtype=float), kept


def split(rows, seed):
    groups = np.asarray([f"{r.get('metadata_plate')}-{r.get('metadata_well')}-{r.get('legend_treatment_label')}" for r in rows])
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * 0.25)))
    test_groups = set(unique[:n_test])
    test = np.asarray([idx for idx, group in enumerate(groups) if group in test_groups], dtype=int)
    train = np.asarray([idx for idx, group in enumerate(groups) if group not in test_groups], dtype=int)
    return train, test


def prep(x_train_raw, x_test_raw, y_train, max_features, x_transform):
    finite = np.isfinite(x_train_raw).any(axis=0)
    x_train_raw = x_train_raw[:, finite]
    x_test_raw = x_test_raw[:, finite]
    med = np.nanmedian(x_train_raw, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    inds = np.where(~np.isfinite(x_train_raw))
    x_train_raw[inds] = np.take(med, inds[1])
    inds = np.where(~np.isfinite(x_test_raw))
    x_test_raw[inds] = np.take(med, inds[1])
    std0 = x_train_raw.std(axis=0)
    keep = std0 > 1e-10
    x_train = x_train_raw[:, keep]
    x_test = x_test_raw[:, keep]
    if x_train.shape[1] == 0:
        return None
    yc = y_train - y_train.mean()
    xc = x_train - x_train.mean(axis=0)
    denom = np.sqrt(np.sum(xc * xc, axis=0) * np.sum(yc * yc))
    corr = np.divide(np.abs(xc.T @ yc), denom, out=np.zeros(x_train.shape[1]), where=denom > 0)
    order = np.argsort(corr)[::-1][: min(max_features, x_train.shape[1])]
    x_train = x_train[:, order]
    x_test = x_test[:, order]
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    if x_transform == "l2":
        x_train = x_train / np.where(np.linalg.norm(x_train, axis=1, keepdims=True) > 1e-12, np.linalg.norm(x_train, axis=1, keepdims=True), 1.0)
        x_test = x_test / np.where(np.linalg.norm(x_test, axis=1, keepdims=True) > 1e-12, np.linalg.norm(x_test, axis=1, keepdims=True), 1.0)
    return x_train, x_test


def knn_predict(x_train, y_train, x_test, k, metric, log_target):
    y_fit = np.log1p(y_train) if log_target else y_train
    preds = []
    for row in x_test:
        dist = pairwise_distances(x_train, row, metric)
        order = np.argsort(dist)[: min(k, len(y_train))]
        weights = 1.0 / np.maximum(dist[order], 1e-9)
        pred = float(np.sum(weights * y_fit[order]) / np.sum(weights))
        preds.append(max(math.expm1(pred), 0.0) if log_target else pred)
    return np.asarray(preds)


def evaluate(rows, cols, target, cohort, feature_set, preprocessing_config):
    x, y, kept = matrix(rows, cols, target, cohort)
    if len(kept) < 12:
        return []
    configs = []
    for max_features in (12, 24, 48, 96, 192):
        for k in (1, 3, 5):
            for metric in ("euclidean", "manhattan", "correlation"):
                for x_transform in ("none", "l2"):
                    for log_target in (False, True):
                        configs.append((max_features, k, metric, x_transform, log_target))
    metrics = []
    for seed in range(10):
        train, test = split(kept, 500 + seed)
        if len(train) < 8 or len(test) < 2:
            continue
        for max_features, k, metric, x_transform, log_target in configs:
            prepared = prep(x[train].copy(), x[test].copy(), y[train], max_features, x_transform)
            if prepared is None:
                continue
            x_train, x_test = prepared
            pred = knn_predict(x_train, y[train], x_test, k, metric, log_target)
            err = y[test] - pred
            rmse = float(np.sqrt(np.mean(err * err)))
            mae = float(np.mean(np.abs(err)))
            ss_res = float(np.sum(err * err))
            ss_tot = float(np.sum((y[test] - y[test].mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
            metrics.append(
                {
                    "target": target,
                    "cohort": cohort,
                    "feature_set": feature_set,
                    "preprocessing_config": preprocessing_config,
                    "model": "weighted_knn",
                    "config": f"k={k},metric={metric},max_features={max_features},x_transform={x_transform},log_target={log_target}",
                    "seed": seed,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                }
            )
    return metrics


def aggregate(rows):
    buckets = {}
    for row in rows:
        key = (row["target"], row["cohort"], row["feature_set"], row["model"], row["config"])
        buckets.setdefault(key, []).append(row)
    out = []
    for key, vals in buckets.items():
        target, cohort, feature_set, model, config = key
        rmse = np.asarray([v["rmse"] for v in vals])
        mae = np.asarray([v["mae"] for v in vals])
        r2 = np.asarray([v["r2"] for v in vals])
        out.append(
            {
                "target": target,
                "cohort": cohort,
                "feature_set": feature_set,
                "model": model,
                "config": config,
                "n_repeats": len(vals),
                "mean_rmse": float(np.nanmean(rmse)),
                "std_rmse": float(np.nanstd(rmse)),
                "mean_mae": float(np.nanmean(mae)),
                "mean_r2": float(np.nanmean(r2)),
            }
        )
    return sorted(out, key=lambda row: (row["target"], row["mean_rmse"]))


def main():
    raman_rows = read_csv(RAMAN_CSV)
    parafac_rows = read_csv(PARAFAC_CSV)
    all_metrics = []
    configs = sorted({row["preprocessing_config"] for row in raman_rows})
    for config in configs:
        rows = [row for row in raman_rows if row["preprocessing_config"] == config]
        cols = feature_cols(rows, ("rp_",))
        for target in TARGETS:
            for cohort in ("all_known", "target_focused"):
                all_metrics.extend(evaluate(rows, cols, target, cohort, f"raman_preprocessed_{config}", config))
    parafac_cols = feature_cols(parafac_rows, ("parafac_score_",))
    for target in TARGETS:
        for cohort in ("all_known", "target_focused"):
            all_metrics.extend(evaluate(parafac_rows, parafac_cols, target, cohort, "eem_parafac_scores", "parafac_rank_selected"))
    for config in configs:
        fused = merge_feature_rows([row for row in raman_rows if row["preprocessing_config"] == config], parafac_rows)
        if not fused:
            continue
        cols = feature_cols(fused, ("rp_", "parafac_score_"))
        for target in TARGETS:
            for cohort in ("all_known", "target_focused"):
                all_metrics.extend(evaluate(fused, cols, target, cohort, f"parafac_raman_fusion_{config}", f"{config}+parafac"))
    if not all_metrics:
        raise RuntimeError("No metrics generated.")
    summary = aggregate(all_metrics)
    best = []
    for target in TARGETS:
        row = next(r for r in summary if r["target"] == target)
        last = PREVIOUS_BEST_RMSE[target]
        row = dict(row)
        row["last_best_rmse"] = last
        row["additional_improvement_vs_last_best_pct"] = 100.0 * (last - row["mean_rmse"]) / last
        row["met_10pct_vs_last_best"] = row["additional_improvement_vs_last_best_pct"] >= 10.0
        best.append(row)
    write_csv(OUT_DIR / "preprocessed_model_search_metrics_by_split.csv", all_metrics, list(all_metrics[0].keys()))
    write_csv(
        OUT_DIR / "preprocessed_model_search_metrics_summary.csv",
        [{k: (f"{v:.8g}" if isinstance(v, float) else v) for k, v in row.items()} for row in summary],
        list(summary[0].keys()),
    )
    write_csv(
        OUT_DIR / "preprocessed_model_best_vs_last.csv",
        [{k: (f"{v:.8g}" if isinstance(v, float) else v) for k, v in row.items()} for row in best],
        list(best[0].keys()),
    )
    for row in best:
        print(
            f"{row['target']}: {row['feature_set']} RMSE={row['mean_rmse']:.5g}; "
            f"additional={row['additional_improvement_vs_last_best_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
