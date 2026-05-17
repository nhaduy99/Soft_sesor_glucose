import csv
import math
import re
from pathlib import Path

import numpy as np

try:
    from scipy import sparse
    from scipy.signal import savgol_filter
    from scipy.sparse.linalg import spsolve
except Exception:
    sparse = None
    savgol_filter = None
    spsolve = None


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw" / "Emilie_SoftSensor"
SOURCE_CSV = ROOT / "features" / "monosaccharide_interpretable_targets.csv"
OUT_CSV = ROOT / "features" / "raman_preprocessed_features.csv"

TARGETS = ("rhamnose_gL", "xylose_gL", "glucose_gL")
WINDOWS = [
    ("rp_w735", 700.0, 770.0),
    ("rp_w905", 870.0, 940.0),
    ("rp_w1156", 1120.0, 1190.0),
    ("rp_w1408", 1370.0, 1445.0),
    ("rp_w1523", 1490.0, 1560.0),
    ("rp_w1878", 1840.0, 1915.0),
]


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return math.nan


def find_raw_file(path_text):
    name = Path(path_text or "").name
    if not name:
        return None
    matches = list(RAW_DIR.rglob(name))
    return matches[0] if matches else None


def parse_targets(label):
    text = (label or "").strip()
    if not text:
        return {}, "unlabelled"
    if text.lower() in {"blank f/2", "milliq", "algae (blank)"}:
        return {target: 0.0 for target in TARGETS}, "blank_or_matrix_blank"
    match = re.search(r"\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)", text)
    if not match:
        return {}, "non_numeric_treatment"
    value = float(match.group(1))
    prefix = text[: match.start()].strip().lower().replace(" ", "")
    if prefix == "rha":
        return {"rhamnose_gL": value, "xylose_gL": 0.0, "glucose_gL": 0.0}, "known_standard_or_spike"
    if prefix == "xyl":
        return {"rhamnose_gL": 0.0, "xylose_gL": value, "glucose_gL": 0.0}, "known_standard_or_spike"
    if prefix == "glu":
        return {"rhamnose_gL": 0.0, "xylose_gL": 0.0, "glucose_gL": value}, "known_standard_or_spike"
    if prefix in {"mmf/2", "mmalgae"}:
        return {target: value for target in TARGETS}, "known_standard_or_spike"
    if prefix == "rha-glu":
        return {"rhamnose_gL": value, "xylose_gL": 0.0, "glucose_gL": value}, "known_standard_or_spike"
    if prefix == "rha-xyl":
        return {"rhamnose_gL": value, "xylose_gL": value, "glucose_gL": 0.0}, "known_standard_or_spike"
    if prefix == "rha-algae":
        return {"rhamnose_gL": value}, "known_standard_or_spike"
    return {}, "unsupported_numeric_treatment"


def parse_raman(path):
    xs, ys = [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                try:
                    xs.append(float(row[0]))
                    ys.append(float(row[1]))
                except ValueError:
                    pass
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    keep = (xs >= 500.0) & (xs <= 2000.0)
    return xs[keep], ys[keep]


def rolling_median(y, window):
    radius = window // 2
    out = np.empty_like(y)
    for i in range(len(y)):
        lo = max(0, i - radius)
        hi = min(len(y), i + radius + 1)
        out[i] = np.median(y[lo:hi])
    return out


def remove_cosmic_spikes(y, window=9, threshold=7.5):
    med = rolling_median(y, window)
    resid = y - med
    mad = np.median(np.abs(resid - np.median(resid)))
    scale = 1.4826 * mad if mad > 1e-12 else np.std(resid)
    if scale <= 1e-12:
        return y.copy()
    out = y.copy()
    mask = np.abs(resid) > threshold * scale
    out[mask] = med[mask]
    return out


def asymmetric_least_squares(y, penalty=None, p=0.01, n_iter=6):
    if sparse is not None and spsolve is not None:
        n = len(y)
        d = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
        penalty_matrix = 1e5 * (d.T @ d)
        weights = np.ones(n)
        for _ in range(n_iter):
            w = sparse.diags(weights, 0, shape=(n, n), format="csc")
            baseline = spsolve(w + penalty_matrix, weights * y)
            weights = p * (y > baseline) + (1.0 - p) * (y <= baseline)
        return np.asarray(baseline, dtype=float)
    n = len(y)
    step = max(1, n // 180)
    if step > 1:
        idx = np.arange(0, n, step)
        if idx[-1] != n - 1:
            idx = np.r_[idx, n - 1]
        small = asymmetric_least_squares(y[idx], penalty=None, p=p, n_iter=n_iter)
        return np.interp(np.arange(n), idx, small)
    if penalty is None:
        d = np.diff(np.eye(n), 2, axis=0)
        penalty = 1e5 * (d.T @ d)
    weights = np.ones(n)
    for _ in range(n_iter):
        w = np.diag(weights)
        baseline = np.linalg.solve(w + penalty, weights * y)
        weights = p * (y > baseline) + (1.0 - p) * (y <= baseline)
    return baseline


def savitzky_golay(y, window=11, poly=3, deriv=0, dx=1.0):
    if window % 2 == 0:
        window += 1
    if savgol_filter is not None:
        return savgol_filter(y, window_length=window, polyorder=poly, deriv=deriv, delta=dx, mode="nearest")
    half = window // 2
    x = np.arange(-half, half + 1, dtype=float)
    a = np.vander(x, poly + 1, increasing=True)
    pinv = np.linalg.pinv(a)
    coeff = pinv[deriv] * math.factorial(deriv) / (dx ** deriv)
    padded = np.pad(y, (half, half), mode="edge")
    out = np.empty_like(y)
    for i in range(len(y)):
        out[i] = np.dot(coeff, padded[i : i + window])
    return out


def snv(y):
    std = np.std(y)
    if std <= 1e-12:
        return y - np.mean(y)
    return (y - np.mean(y)) / std


def area_normalize(x, y):
    area = np.trapezoid(np.abs(y), x)
    if area <= 1e-12:
        return y
    return y / area


def preprocess(x, y, penalty, derivative=0, area_norm=False):
    y1 = remove_cosmic_spikes(y)
    baseline = asymmetric_least_squares(y1, penalty)
    corrected = y1 - baseline
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    smooth = savitzky_golay(corrected, window=11, poly=3, deriv=derivative, dx=dx)
    normalized = snv(smooth)
    if area_norm:
        normalized = area_normalize(x, normalized)
    return normalized


def summarize_windows(x, y):
    out = {}
    for name, lo, hi in WINDOWS:
        vals = y[(x >= lo) & (x <= hi)]
        if len(vals) == 0:
            out[f"{name}_mean"] = ""
            out[f"{name}_max"] = ""
            out[f"{name}_area"] = ""
        else:
            out[f"{name}_mean"] = f"{float(np.mean(vals)):.8g}"
            out[f"{name}_max"] = f"{float(np.max(vals)):.8g}"
            out[f"{name}_area"] = f"{float(np.trapezoid(vals, dx=1.0)):.8g}"
    if out.get("rp_w905_mean") not in {"", "0"} and out.get("rp_w735_mean") not in {"", "0"}:
        out["rp_ratio_905_735"] = f"{safe_float(out['rp_w905_mean']) / safe_float(out['rp_w735_mean']):.8g}"
    else:
        out["rp_ratio_905_735"] = ""
    return out


def main():
    source_rows = read_csv(SOURCE_CSV)
    configs = [
        ("als_sg0_snv", 0, False),
        ("als_sg1_snv", 1, False),
        ("als_sg2_snv", 2, False),
        ("als_sg0_snv_area", 0, True),
        ("als_sg1_snv_area", 1, True),
    ]
    out_rows = []
    path_cache = {}
    penalty_cache = {}
    feature_cache = {}
    for row in source_rows:
        targets, target_source = parse_targets(row.get("legend_treatment_label", ""))
        if not targets:
            continue
        path = find_raw_file(row.get("raman_file"))
        if not path:
            continue
        if path not in path_cache:
            x, y = parse_raman(path)
            path_cache[path] = (x, y)
            penalty_cache[path] = None
        x, y = path_cache[path]
        if len(x) < 20:
            continue
        for config_name, derivative, area_norm in configs:
            cache_key = (path, config_name)
            if cache_key not in feature_cache:
                processed = preprocess(x, y, penalty_cache[path], derivative=derivative, area_norm=area_norm)
                feat = summarize_windows(x, processed)
                for idx in range(0, len(processed), 10):
                    feat[f"rp_full_{int(round(x[idx]))}"] = f"{float(processed[idx]):.8g}"
                feature_cache[cache_key] = feat
            out = {k: row.get(k, "") for k in row.keys() if not k.startswith("eem_") and not k.startswith("r735_")}
            out["preprocessing_config"] = config_name
            out["raman_processed_points"] = str(len(processed))
            out["target_source"] = target_source
            for target in TARGETS:
                out[target] = "" if target not in targets else f"{targets[target]:.8g}"
            out.update(feature_cache[cache_key])
            out_rows.append(out)
    fieldnames = []
    seen = set()
    for row in out_rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    write_csv(OUT_CSV, out_rows, fieldnames)
    print(f"Wrote {OUT_CSV}")
    print(f"Rows: {len(out_rows)}")
    print(f"Configurations: {', '.join(c[0] for c in configs)}")


if __name__ == "__main__":
    main()
