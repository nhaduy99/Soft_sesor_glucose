import csv
import math
import os
import re
from pathlib import Path

import numpy as np

try:
    import tensorly as tl
    from tensorly.decomposition import non_negative_parafac
except Exception:
    tl = None
    non_negative_parafac = None


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw" / "Emilie_SoftSensor"
SOURCE_CSV = ROOT / "features" / "monosaccharide_interpretable_targets.csv"
EXCLUDE_RHA5 = os.environ.get("EXCLUDE_RHA5", "").strip().lower() in {"1", "true", "yes"}
OUT_DIR = ROOT / os.environ.get(
    "EEM_PARAFAC_OUT_DIR",
    "features/eem_parafac_exclude_rha5" if EXCLUDE_RHA5 else "features/eem_parafac",
)
SCORES_CSV = ROOT / os.environ.get(
    "EEM_PARAFAC_SCORES_CSV",
    "features/eem_parafac_scores_exclude_rha5.csv" if EXCLUDE_RHA5 else "features/eem_parafac_scores.csv",
)
SUMMARY_CSV = OUT_DIR / "parafac_rank_summary.csv"
TARGETS = ("rhamnose_gL", "xylose_gL", "glucose_gL")
SCATTER_PRIMARY_NM = 20.0
SCATTER_SECOND_ORDER_NM = 25.0


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


def is_excluded_rha5(row):
    if not EXCLUDE_RHA5:
        return False
    label = (row.get("legend_treatment_label") or "").strip().lower().replace(" ", "")
    targets, _ = parse_targets(row.get("legend_treatment_label", ""))
    return label == "rha(5)" and targets.get("rhamnose_gL") == 5.0


def parse_eem(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    emission = [safe_float(v) for v in rows[0][1:]]
    excitation, matrix = [], []
    for row in rows[1:]:
        ex = safe_float(row[0])
        excitation.append(ex)
        vals = []
        for j, cell in enumerate(row[1:]):
            em = emission[j]
            is_over = str(cell).strip().upper() == "OVER"
            primary_scatter = em <= ex + SCATTER_PRIMARY_NM or abs(em - ex) <= SCATTER_PRIMARY_NM
            second_order_scatter = abs(em - 2.0 * ex) <= SCATTER_SECOND_ORDER_NM
            if is_over or primary_scatter or second_order_scatter:
                vals.append(math.nan)
            else:
                vals.append(safe_float(cell))
        matrix.append(vals)
    return np.asarray(excitation), np.asarray(emission), np.asarray(matrix, dtype=float)


def load_eem_cube():
    rows = [row for row in read_csv(SOURCE_CSV) if not is_excluded_rha5(row)]
    samples, matrices = [], []
    excitation = emission = None
    for row in rows:
        path = find_raw_file(row.get("eem_file"))
        if not path:
            continue
        ex, em, mat = parse_eem(path)
        if excitation is None:
            excitation, emission = ex, em
        if mat.shape != (len(excitation), len(emission)):
            continue
        samples.append(row)
        matrices.append(mat)
    cube = np.asarray(matrices, dtype=float)
    med = np.nanmedian(cube, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    inds = np.where(~np.isfinite(cube))
    cube[inds] = med[inds[1], inds[2]]
    cube = np.maximum(cube, 0.0)
    scale = np.nanmax(cube)
    if scale > 0:
        cube = cube / scale
    return samples, excitation, emission, cube


def khatri_rao(a, b):
    cols = []
    for r in range(a.shape[1]):
        cols.append(np.kron(a[:, r], b[:, r]))
    return np.column_stack(cols)


def normalize_columns(mat):
    norms = np.linalg.norm(mat, axis=0)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return mat / norms, norms


def reconstruct(a, b, c, weights):
    out = np.zeros((a.shape[0], b.shape[0], c.shape[0]))
    for r in range(len(weights)):
        out += weights[r] * np.einsum("i,j,k->ijk", a[:, r], b[:, r], c[:, r])
    return out


def cp_als(x, rank, n_iter=80, seed=0):
    if non_negative_parafac is not None:
        tl.set_backend("numpy")
        weights, factors = non_negative_parafac(
            x,
            rank=rank,
            n_iter_max=n_iter,
            init="random",
            random_state=seed,
            normalize_factors=True,
            tol=1e-7,
        )
        a, b, c = [np.asarray(factor, dtype=float) for factor in factors]
        weights = np.asarray(weights, dtype=float)
        rec = reconstruct(a, b, c, weights)
        error = float(np.linalg.norm(x - rec) / max(np.linalg.norm(x), 1e-12))
        scores = a * weights
        return {"scores": scores, "excitation": b, "emission": c, "weights": weights, "error": error}
    rng = np.random.default_rng(seed)
    i, j, k = x.shape
    a = rng.random((i, rank)) + 0.1
    b = rng.random((j, rank)) + 0.1
    c = rng.random((k, rank)) + 0.1
    weights = np.ones(rank)
    x1 = x.reshape(i, j * k)
    x2 = np.transpose(x, (1, 0, 2)).reshape(j, i * k)
    x3 = np.transpose(x, (2, 0, 1)).reshape(k, i * j)
    for _ in range(n_iter):
        kr = khatri_rao(c, b)
        a = x1 @ kr @ np.linalg.pinv((b.T @ b) * (c.T @ c))
        a = np.maximum(a, 0.0)
        a, na = normalize_columns(a)

        kr = khatri_rao(c, a)
        b = x2 @ kr @ np.linalg.pinv((a.T @ a) * (c.T @ c))
        b = np.maximum(b, 0.0)
        b, nb = normalize_columns(b)

        kr = khatri_rao(b, a)
        c = x3 @ kr @ np.linalg.pinv((a.T @ a) * (b.T @ b))
        c = np.maximum(c, 0.0)
        c, nc = normalize_columns(c)
        weights = na * nb * nc
        weights = np.where(weights > 1e-12, weights, 1.0)
    rec = reconstruct(a, b, c, weights)
    error = float(np.linalg.norm(x - rec) / max(np.linalg.norm(x), 1e-12))
    scores = a * weights
    return {"scores": scores, "excitation": b, "emission": c, "weights": weights, "error": error}


def component_similarity(model_a, model_b):
    sims = []
    for r in range(model_a["excitation"].shape[1]):
        vec_a = np.r_[model_a["excitation"][:, r], model_a["emission"][:, r]]
        best = 0.0
        for s in range(model_b["excitation"].shape[1]):
            vec_b = np.r_[model_b["excitation"][:, s], model_b["emission"][:, s]]
            denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
            if denom > 1e-12:
                best = max(best, abs(float(vec_a @ vec_b / denom)))
        sims.append(best)
    return float(np.mean(sims)) if sims else 0.0


def ridge_cv_score(scores, rows, target):
    y = []
    keep = []
    for idx, row in enumerate(rows):
        targets, _ = parse_targets(row.get("legend_treatment_label", ""))
        if target in targets:
            y.append(targets[target])
            keep.append(idx)
    if len(keep) < 12:
        return math.nan
    x = scores[keep]
    y = np.asarray(y, dtype=float)
    rmses = []
    for seed in range(5):
        rng = np.random.default_rng(200 + seed)
        idx = np.arange(len(y))
        rng.shuffle(idx)
        n_test = max(2, int(round(len(idx) * 0.25)))
        test = idx[:n_test]
        train = idx[n_test:]
        mean = x[train].mean(axis=0)
        std = x[train].std(axis=0)
        std = np.where(std > 1e-12, std, 1.0)
        xt = (x[train] - mean) / std
        xv = (x[test] - mean) / std
        coef = np.linalg.pinv(xt.T @ xt + np.eye(xt.shape[1])) @ xt.T @ y[train]
        intercept = y[train].mean() - xt.mean(axis=0) @ coef
        pred = intercept + xv @ coef
        rmses.append(float(np.sqrt(np.mean((y[test] - pred) ** 2))))
    return float(np.mean(rmses))


def svg_polyline(xs, ys, color):
    if len(xs) == 0:
        return ""
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if y1 <= y0:
        y1 = y0 + 1
    pts = []
    for x, y in zip(xs, ys):
        px = 70 + (x - x0) / (x1 - x0) * 420
        py = 260 - (y - y0) / (y1 - y0) * 210
        pts.append(f"{px:.1f},{py:.1f}")
    return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2"/>'


def write_loading_plots(rank, model, excitation, emission):
    colors = ["#2f7d7e", "#7a4e9f", "#b5651d", "#2d5f9a", "#6b7280", "#11845b", "#a43f5f", "#7c5c00"]
    for mode, wavelengths, loadings in (
        ("excitation", excitation, model["excitation"]),
        ("emission", emission, model["emission"]),
    ):
        lines = []
        legend = []
        for r in range(rank):
            color = colors[r % len(colors)]
            lines.append(svg_polyline(wavelengths, loadings[:, r], color))
            legend.append(f'<text x="520" y="{62 + r * 18}" font-size="12" fill="{color}">Component {r+1}</text>')
        svg = f"""<svg viewBox="0 0 680 330" xmlns="http://www.w3.org/2000/svg">
<rect x="0" y="0" width="680" height="330" fill="#fff"/>
<text x="340" y="28" text-anchor="middle" font-size="18" font-weight="700">PARAFAC rank {rank}: {mode} loadings</text>
<rect x="70" y="50" width="420" height="210" fill="#fff" stroke="#17202a"/>
{''.join(lines)}
{''.join(legend)}
<text x="280" y="302" text-anchor="middle" font-size="13">Wavelength (nm)</text>
<text x="26" y="160" text-anchor="middle" transform="rotate(-90 26 160)" font-size="13">Loading</text>
</svg>"""
        (OUT_DIR / f"rank{rank}_{mode}_loadings.svg").write_text(svg, encoding="utf-8")


def write_component_maps(rank, model, excitation, emission):
    for r in range(rank):
        comp = np.outer(model["excitation"][:, r], model["emission"][:, r])
        lo, hi = float(np.min(comp)), float(np.max(comp))
        cells = []
        for i in range(comp.shape[0]):
            for j in range(comp.shape[1]):
                t = (comp[i, j] - lo) / (hi - lo) if hi > lo else 0
                red = int(245 - 190 * t)
                green = int(248 - 110 * t)
                blue = int(251 - 10 * t)
                cells.append(f'<rect x="{72+j*20}" y="{48+i*18}" width="20.2" height="18.2" fill="rgb({red},{green},{blue})"/>')
        svg = f"""<svg viewBox="0 0 560 380" xmlns="http://www.w3.org/2000/svg">
<rect x="0" y="0" width="560" height="380" fill="#fff"/>
<text x="280" y="28" text-anchor="middle" font-size="18" font-weight="700">PARAFAC rank {rank}, component {r+1}</text>
<rect x="72" y="48" width="{comp.shape[1]*20}" height="{comp.shape[0]*18}" fill="#fff" stroke="#17202a"/>
{''.join(cells)}
<text x="260" y="350" text-anchor="middle" font-size="13">Emission wavelength axis</text>
<text x="24" y="185" text-anchor="middle" transform="rotate(-90 24 185)" font-size="13">Excitation wavelength axis</text>
</svg>"""
        (OUT_DIR / f"rank{rank}_component{r+1}_map.svg").write_text(svg, encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, excitation, emission, cube = load_eem_cube()
    if len(rows) < 10:
        raise RuntimeError("Not enough EEM samples for PARAFAC.")
    summary = []
    models = {}
    for rank in range(2, 9):
        model = cp_als(cube, rank=rank, n_iter=80, seed=rank)
        half = len(rows) // 2
        model_a = cp_als(cube[:half], rank=rank, n_iter=60, seed=rank + 100)
        model_b = cp_als(cube[half:], rank=rank, n_iter=60, seed=rank + 200)
        stability = component_similarity(model_a, model_b)
        pred_rmses = [ridge_cv_score(model["scores"], rows, target) for target in TARGETS]
        pred_score = np.nanmean(pred_rmses)
        summary.append(
            {
                "rank": rank,
                "parafac_backend": "tensorly_non_negative_parafac" if non_negative_parafac is not None else "numpy_cp_als",
                "scatter_mask": f"primary_nm={SCATTER_PRIMARY_NM};second_order_nm={SCATTER_SECOND_ORDER_NM}",
                "reconstruction_error": f"{model['error']:.8g}",
                "split_half_stability": f"{stability:.8g}",
                "prediction_rmse_mean": f"{pred_score:.8g}",
                "rhamnose_score_rmse": f"{pred_rmses[0]:.8g}",
                "xylose_score_rmse": f"{pred_rmses[1]:.8g}",
                "glucose_score_rmse": f"{pred_rmses[2]:.8g}",
            }
        )
        models[rank] = model
    def selection_key(item):
        rank = item["rank"]
        error = safe_float(item["reconstruction_error"])
        stability = safe_float(item["split_half_stability"])
        pred = safe_float(item["prediction_rmse_mean"])
        return pred + 0.35 * error - 0.15 * stability + 0.01 * rank
    selected = min(summary, key=selection_key)
    selected_rank = int(selected["rank"])
    for row in summary:
        row["selected_rank"] = "True" if int(row["rank"]) == selected_rank else "False"
    write_csv(SUMMARY_CSV, summary, list(summary[0].keys()))

    model = models[selected_rank]
    out_rows = []
    for idx, src in enumerate(rows):
        targets, target_source = parse_targets(src.get("legend_treatment_label", ""))
        out = {k: src.get(k, "") for k in src.keys() if not k.startswith("eem_") and not k.startswith("raman_")}
        out["parafac_selected_rank"] = str(selected_rank)
        out["target_source"] = target_source
        for target in TARGETS:
            out[target] = "" if target not in targets else f"{targets[target]:.8g}"
        for comp in range(selected_rank):
            out[f"parafac_score_c{comp+1}"] = f"{float(model['scores'][idx, comp]):.8g}"
        out_rows.append(out)
    fieldnames = []
    seen = set()
    for row in out_rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    write_csv(SCORES_CSV, out_rows, fieldnames)

    for rank, rank_model in models.items():
        if rank == selected_rank:
            write_loading_plots(rank, rank_model, excitation, emission)
            write_component_maps(rank, rank_model, excitation, emission)
            np.savetxt(OUT_DIR / f"rank{rank}_sample_scores.csv", rank_model["scores"], delimiter=",")
            np.savetxt(OUT_DIR / f"rank{rank}_excitation_loadings.csv", rank_model["excitation"], delimiter=",")
            np.savetxt(OUT_DIR / f"rank{rank}_emission_loadings.csv", rank_model["emission"], delimiter=",")
    print(f"Wrote {SCORES_CSV}")
    print(f"Selected PARAFAC rank: {selected_rank}")
    print(f"Rank summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
