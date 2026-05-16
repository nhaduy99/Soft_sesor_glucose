import csv
import math
from pathlib import Path


ROOT = Path(
    r"C:\Users\nhadu\OneDrive - UTS\C3_UTS\Soft_Sensor_Loc\Emilie data\Raw_data\Emilie_SoftSensor"
)
CODEX_DIR = ROOT / "Codex_inventory"
INVENTORY_CSV = CODEX_DIR / "eem_raman_hplc_inventory_enriched.csv"
OUT_DIR = CODEX_DIR / "features"


RAMAN_WINDOWS = [
    ("r735", 700.0, 770.0),
    ("r905", 870.0, 940.0),
    ("r1156", 1120.0, 1190.0),
    ("r1408", 1370.0, 1445.0),
    ("r1523", 1490.0, 1560.0),
    ("r1878", 1840.0, 1915.0),
]

EEM_HOTSPOTS = [
    ("eem_h1_ex380_em400", 380.0, 400.0),
    ("eem_h2_ex380_em420", 380.0, 420.0),
    ("eem_h3_ex400_em420", 400.0, 420.0),
    ("eem_h4_ex400_em440", 400.0, 440.0),
    ("eem_h5_ex420_em460", 420.0, 460.0),
    ("eem_h6_ex440_em480", 440.0, 480.0),
]


def safe_float(text):
    try:
        return float(text)
    except Exception:
        return None


def mean(values):
    return sum(values) / len(values) if values else None


def max_value(values):
    return max(values) if values else None


def sum_value(values):
    return sum(values) if values else None


def normalize_header_key(row):
    fixed = dict(row)
    if "ï»¿\"group_code\"" in fixed:
        fixed["group_code"] = fixed.pop("ï»¿\"group_code\"")
    if '\ufeff"group_code"' in fixed:
        fixed["group_code"] = fixed.pop('\ufeff"group_code"')
    return fixed


def load_inventory():
    with INVENTORY_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [normalize_header_key(row) for row in reader]


def parse_eem(path_text):
    with open(path_text, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    emission = [safe_float(v) for v in rows[0][1:]]
    excitation = [safe_float(r[0]) for r in rows[1:]]
    values = []
    over_mask = []
    for row in rows[1:]:
        val_row = []
        over_row = []
        for cell in row[1:]:
            if cell == "OVER":
                val_row.append(None)
                over_row.append(True)
            else:
                val_row.append(float(cell))
                over_row.append(False)
        values.append(val_row)
        over_mask.append(over_row)
    return emission, excitation, values, over_mask


def parse_raman(path_text):
    xs = []
    ys = []
    with open(path_text, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            xs.append(float(row[0]))
            ys.append(float(row[1]))
    return xs, ys


def nearest_index(values, target):
    best_idx = 0
    best_dist = abs(values[0] - target)
    for idx, value in enumerate(values[1:], start=1):
        dist = abs(value - target)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def cleaned_eem_values(emission, excitation, values, over_mask):
    cleaned = []
    for i, ex in enumerate(excitation):
        row = []
        for j, em in enumerate(emission):
            value = values[i][j]
            if over_mask[i][j]:
                row.append(None)
            elif em <= ex + 20.0:
                row.append(None)
            else:
                row.append(value)
        cleaned.append(row)
    return cleaned


def extract_eem_features(path_text):
    emission, excitation, values, over_mask = parse_eem(path_text)
    cleaned = cleaned_eem_values(emission, excitation, values, over_mask)

    valid = [v for row in cleaned for v in row if v is not None]
    raw_valid = [v for row in values for v in row if v is not None]
    over_fraction = sum(1 for row in over_mask for cell in row if cell) / max(1, len(over_mask) * len(over_mask[0]))

    feat = {
        "eem_available": "True",
        "eem_raw_mean": raw_valid and f"{mean(raw_valid):.6f}" or "",
        "eem_raw_max": raw_valid and f"{max_value(raw_valid):.6f}" or "",
        "eem_clean_mean": valid and f"{mean(valid):.6f}" or "",
        "eem_clean_max": valid and f"{max_value(valid):.6f}" or "",
        "eem_over_fraction_total": f"{over_fraction:.6f}",
        "eem_clean_valid_fraction": f"{(len(valid) / max(1, len(values) * len(values[0]))):.6f}",
    }

    for name, ex_target, em_target in EEM_HOTSPOTS:
        i = nearest_index(excitation, ex_target)
        j = nearest_index(emission, em_target)
        value = cleaned[i][j]
        feat[name] = "" if value is None else f"{value:.6f}"

    # Full cleaned vector for later PLS-style modeling.
    vector = []
    names = []
    for i, ex in enumerate(excitation):
        for j, em in enumerate(emission):
            if em <= ex + 20.0:
                continue
            names.append(f"eem_ex{int(ex)}_em{int(em)}")
            value = cleaned[i][j]
            vector.append("" if value is None else f"{value:.6f}")
    feat["_eem_vector_names"] = names
    feat["_eem_vector_values"] = vector
    return feat


def integrate_window(xs, ys, lo, hi):
    vals = [y for x, y in zip(xs, ys) if lo <= x <= hi]
    return {
        "mean": mean(vals),
        "max": max_value(vals),
        "sum": sum_value(vals),
    }


def extract_raman_features(path_text):
    xs, ys = parse_raman(path_text)
    cropped = [(x, y) for x, y in zip(xs, ys) if 500.0 <= x <= 2000.0]
    cxs = [x for x, _ in cropped]
    cys = [y for _, y in cropped]
    feat = {
        "raman_available": "True",
        "raman_crop_mean": cys and f"{mean(cys):.6f}" or "",
        "raman_crop_max": cys and f"{max_value(cys):.6f}" or "",
        "raman_crop_sum": cys and f"{sum_value(cys):.6f}" or "",
    }

    peak_means = {}
    for name, lo, hi in RAMAN_WINDOWS:
        stats = integrate_window(cxs, cys, lo, hi)
        feat[f"{name}_mean"] = "" if stats["mean"] is None else f"{stats['mean']:.6f}"
        feat[f"{name}_max"] = "" if stats["max"] is None else f"{stats['max']:.6f}"
        feat[f"{name}_sum"] = "" if stats["sum"] is None else f"{stats['sum']:.6f}"
        peak_means[name] = stats["mean"]

    if peak_means["r1523"] is not None and peak_means["r1156"] not in (None, 0):
        feat["raman_ratio_1523_1156"] = f"{(peak_means['r1523'] / peak_means['r1156']):.6f}"
    else:
        feat["raman_ratio_1523_1156"] = ""
    if peak_means["r905"] is not None and peak_means["r735"] not in (None, 0):
        feat["raman_ratio_905_735"] = f"{(peak_means['r905'] / peak_means['r735']):.6f}"
    else:
        feat["raman_ratio_905_735"] = ""

    feat["_raman_vector_names"] = [f"raman_{int(round(x))}" for x in cxs]
    feat["_raman_vector_values"] = [f"{y:.6f}" for y in cys]
    return feat


def merge_features(row, eem_feat, raman_feat):
    out = dict(row)
    if eem_feat:
        out.update({k: v for k, v in eem_feat.items() if not k.startswith("_")})
    else:
        out["eem_available"] = "False"
    if raman_feat:
        out.update({k: v for k, v in raman_feat.items() if not k.startswith("_")})
    else:
        out["raman_available"] = "False"
    return out


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def union_fieldnames(rows):
    seen = []
    known = set()
    for row in rows:
        for key in row.keys():
            if key not in known:
                known.add(key)
                seen.append(key)
    return seen


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = load_inventory()
    interpretable_rows = []
    full_rows = []
    eem_vector_names = None
    raman_vector_names = None

    for row in inventory:
        eem_feat = None
        raman_feat = None
        eem_path = row.get("eem_file", "").strip()
        raman_path = row.get("raman_file", "").strip()
        if eem_path:
            eem_feat = extract_eem_features(eem_path)
            if eem_vector_names is None:
                eem_vector_names = eem_feat["_eem_vector_names"]
        if raman_path:
            raman_feat = extract_raman_features(raman_path)
            if raman_vector_names is None:
                raman_vector_names = raman_feat["_raman_vector_names"]

        merged = merge_features(row, eem_feat, raman_feat)
        interpretable_rows.append(merged)

        full_row = dict(merged)
        if eem_vector_names:
            values = eem_feat["_eem_vector_values"] if eem_feat else [""] * len(eem_vector_names)
            for name, value in zip(eem_vector_names, values):
                full_row[name] = value
        if raman_vector_names:
            values = raman_feat["_raman_vector_values"] if raman_feat else [""] * len(raman_vector_names)
            for name, value in zip(raman_vector_names, values):
                full_row[name] = value
        full_rows.append(full_row)

    interpretable_fields = union_fieldnames(interpretable_rows)
    full_fields = union_fieldnames(full_rows)

    write_csv(OUT_DIR / "rhamnose_interpretable_features.csv", interpretable_rows, interpretable_fields)
    write_csv(OUT_DIR / "rhamnose_full_feature_matrix.csv", full_rows, full_fields)

    summary_rows = [
        {"metric": "rows", "value": str(len(inventory)), "note": ""},
        {"metric": "rows_with_eem", "value": str(sum(1 for r in inventory if r.get("eem_file", "").strip())), "note": ""},
        {"metric": "rows_with_raman", "value": str(sum(1 for r in inventory if r.get("raman_file", "").strip())), "note": ""},
        {"metric": "eem_clean_hotspot_features", "value": str(len(EEM_HOTSPOTS)), "note": "Interpretable masked EEM hotspot summaries"},
        {"metric": "raman_peak_windows", "value": str(len(RAMAN_WINDOWS)), "note": "Interpretable Raman windows around detected mean-spectrum peaks"},
        {"metric": "eem_vector_columns", "value": str(len(eem_vector_names or [])), "note": "Full cleaned EEM vector columns"},
        {"metric": "raman_vector_columns", "value": str(len(raman_vector_names or [])), "note": "Full cropped Raman vector columns"},
    ]
    write_csv(OUT_DIR / "rhamnose_feature_summary.csv", summary_rows, ["metric", "value", "note"])
    print(f"Output directory: {OUT_DIR}")
    print(f"Rows exported: {len(inventory)}")


if __name__ == "__main__":
    build()
