import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(
    r"C:\Users\nhadu\OneDrive - UTS\C3_UTS\Soft_Sensor_Loc\Emilie data\Raw_data\Emilie_SoftSensor"
)
FEATURES_DIR = ROOT / "Codex_inventory" / "features"
INPUT_CSV = FEATURES_DIR / "rhamnose_interpretable_features.csv"
OUT_DIR = ROOT / "Codex_inventory" / "unsupervised"


RAMAN_COLS = [
    "raman_crop_mean",
    "raman_crop_max",
    "raman_crop_sum",
    "r735_mean",
    "r735_max",
    "r735_sum",
    "r905_mean",
    "r905_max",
    "r905_sum",
    "r1156_mean",
    "r1156_max",
    "r1156_sum",
    "r1408_mean",
    "r1408_max",
    "r1408_sum",
    "r1523_mean",
    "r1523_max",
    "r1523_sum",
    "r1878_mean",
    "r1878_max",
    "r1878_sum",
    "raman_ratio_1523_1156",
    "raman_ratio_905_735",
]

EEM_COLS = [
    "eem_raw_mean",
    "eem_raw_max",
    "eem_clean_mean",
    "eem_clean_max",
    "eem_over_fraction_total",
    "eem_clean_valid_fraction",
    "eem_h1_ex380_em400",
    "eem_h2_ex380_em420",
    "eem_h3_ex400_em420",
    "eem_h4_ex400_em440",
    "eem_h5_ex420_em460",
    "eem_h6_ex440_em480",
]


def safe_float(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def load_rows():
    with INPUT_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_matrix(rows, columns, require_col):
    kept_rows = []
    matrix = []
    for row in rows:
        if row.get(require_col) != "True":
            continue
        vals = []
        for col in columns:
            x = safe_float(row.get(col))
            vals.append(np.nan if x is None or math.isnan(x) or math.isinf(x) else x)
        kept_rows.append(row)
        matrix.append(vals)
    return kept_rows, np.asarray(matrix, dtype=float)


def impute_column_means(X):
    X = X.copy()
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    return X


def standardize(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Z = (X - mu) / sigma
    return Z, mu, sigma


def pca(X, n_components=2):
    Z, mu, sigma = standardize(X)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    scores = Z @ Vt.T[:, :n_components]
    explained = (S ** 2) / max(1, (len(Z) - 1))
    ratio = explained / explained.sum()
    loadings = Vt[:n_components, :]
    return {
        "scores": scores,
        "explained_ratio": ratio[:n_components],
        "loadings": loadings,
        "mean": mu,
        "std": sigma,
    }


def kmeans(X, k=3, n_iter=50):
    # deterministic farthest-point initialization
    centers = [X[0]]
    while len(centers) < k:
        d2 = []
        for row in X:
            d2.append(min(np.sum((row - c) ** 2) for c in centers))
        centers.append(X[int(np.argmax(d2))])
    centers = np.asarray(centers, dtype=float)

    labels = np.zeros(len(X), dtype=int)
    for _ in range(n_iter):
        dists = np.stack([np.sum((X - c) ** 2, axis=1) for c in centers], axis=1)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                centers[i] = X[mask].mean(axis=0)
    inertia = float(sum(np.sum((X[labels == i] - centers[i]) ** 2) for i in range(k) if np.any(labels == i)))
    return labels, centers, inertia


def top_loadings(columns, loading_vector, top_n=6):
    pairs = sorted(((abs(v), v, c) for c, v in zip(columns, loading_vector)), reverse=True)
    return [(c, v) for _, v, c in pairs[:top_n]]


def svg_header(width, height):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #111; }",
        ".title { font-size: 22px; font-weight: bold; }",
        ".subtitle { font-size: 12px; fill: #444; }",
        ".axis { stroke: #222; stroke-width: 1.2; }",
        ".grid { stroke: #ddd; stroke-width: 1; }",
        ".small { font-size: 11px; }",
        ".label { font-size: 12px; }",
        ".anno { font-size: 11px; fill: #7a1010; font-weight: bold; }",
        "</style>",
    ]


def write_svg(path, lines):
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def palette(i):
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    return colors[i % len(colors)]


def render_scatter_svg(path, title, subtitle, scores, labels, explained, sample_labels):
    width, height = 1100, 760
    x0, y0, w, h = 90, 110, 650, 520
    xs = scores[:, 0]
    ys = scores[:, 1]
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    padx = (xmax - xmin) * 0.08 if xmax > xmin else 1.0
    pady = (ymax - ymin) * 0.08 if ymax > ymin else 1.0
    xmin -= padx
    xmax += padx
    ymin -= pady
    ymax += pady

    lines = svg_header(width, height)
    lines += [
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        f'<text x="90" y="50" class="title">{title}</text>',
        f'<text x="90" y="75" class="subtitle">{subtitle}</text>',
    ]
    for frac in range(0, 6):
        gy = y0 + h - frac * h / 5
        gv = ymin + frac * (ymax - ymin) / 5
        lines.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x0+w}" y2="{gy:.1f}" class="grid"/>')
        lines.append(f'<text x="{x0-8}" y="{gy+4:.1f}" text-anchor="end" class="small">{gv:.1f}</text>')
    for frac in range(0, 6):
        gx = x0 + frac * w / 5
        gv = xmin + frac * (xmax - xmin) / 5
        lines.append(f'<line x1="{gx:.1f}" y1="{y0}" x2="{gx:.1f}" y2="{y0+h}" class="grid"/>')
        lines.append(f'<text x="{gx:.1f}" y="{y0+h+20}" text-anchor="middle" class="small">{gv:.1f}</text>')
    lines += [
        f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" class="axis"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}" class="axis"/>',
        f'<text x="{x0+w/2:.1f}" y="{y0+h+45}" text-anchor="middle" class="label">PC1 ({explained[0]*100:.1f}% variance)</text>',
        f'<text x="28" y="{y0+h/2:.1f}" transform="rotate(-90 28,{y0+h/2:.1f})" text-anchor="middle" class="label">PC2 ({explained[1]*100:.1f}% variance)</text>',
    ]
    for x, y, lab in zip(xs, ys, labels):
        px = x0 + (x - xmin) / (xmax - xmin) * w
        py = y0 + h - (y - ymin) / (ymax - ymin) * h
        lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.6" fill="{palette(int(lab))}" fill-opacity="0.72"/>')

    # annotate representative points with largest distance from origin
    d = np.sum(scores[:, :2] ** 2, axis=1)
    chosen = np.argsort(d)[-8:]
    for idx in chosen:
        x, y = xs[idx], ys[idx]
        px = x0 + (x - xmin) / (xmax - xmin) * w
        py = y0 + h - (y - ymin) / (ymax - ymin) * h
        lines.append(f'<text x="{px+6:.2f}" y="{py-6:.2f}" class="anno">{sample_labels[idx]}</text>')

    legend_x = 790
    legend_y = 140
    for c in sorted(set(labels)):
        lines.append(f'<circle cx="{legend_x}" cy="{legend_y + c*24}" r="6" fill="{palette(int(c))}"/>')
        lines.append(f'<text x="{legend_x+14}" y="{legend_y + c*24 + 4}" class="small">Cluster {int(c)}</text>')
    write_svg(path, lines)


def render_bar_svg(path, title, subtitle, items):
    width, height = 1100, 620
    x0, y0, w, h = 110, 110, 850, 380
    maxv = max(abs(v) for _, v in items) if items else 1.0
    lines = svg_header(width, height)
    lines += [
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        f'<text x="90" y="50" class="title">{title}</text>',
        f'<text x="90" y="75" class="subtitle">{subtitle}</text>',
    ]
    n = len(items)
    bw = w / max(1, n)
    zero_y = y0 + h / 2
    lines.append(f'<line x1="{x0}" y1="{zero_y:.1f}" x2="{x0+w}" y2="{zero_y:.1f}" class="axis"/>')
    for i, (name, val) in enumerate(items):
        x = x0 + i * bw + 10
        bar_h = (abs(val) / maxv) * (h * 0.42)
        y = zero_y - bar_h if val >= 0 else zero_y
        color = "#1f77b4" if val >= 0 else "#d62728"
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw-20:.1f}" height="{bar_h:.1f}" fill="{color}" opacity="0.8"/>')
        lines.append(f'<text x="{x + (bw-20)/2:.1f}" y="{zero_y + 18 if val >= 0 else zero_y - 8:.1f}" text-anchor="middle" class="small">{name}</text>')
        lines.append(f'<text x="{x + (bw-20)/2:.1f}" y="{y - 6 if val >= 0 else y + bar_h + 14:.1f}" text-anchor="middle" class="small">{val:.2f}</text>')
    write_svg(path, lines)


def summarize_clusters(rows, labels):
    out = defaultdict(lambda: {"sample_set": Counter(), "measurement": Counter(), "label": Counter()})
    for row, label in zip(rows, labels):
        bucket = out[int(label)]
        bucket["sample_set"][row.get("sample_set", "")] += 1
        bucket["measurement"][row.get("matching_measurements", "")] += 1
        bucket["label"][row.get("legend_treatment_label", "")] += 1
    return out


def read_inline_svg(path):
    text = path.read_text(encoding="utf-8")
    start = text.find("<svg")
    return text[start:] if start >= 0 else text


def write_report(report_path, sections):
    rows = []
    for sec in sections:
        rows.append(f"<div class='card'><h2>{sec['title']}</h2><p>{sec['text']}</p>{sec.get('svg','')}{sec.get('extra','')}</div>")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Unsupervised Feature Exploration</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f7f7f7; color: #111; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 32px; }}
    .card {{ background: #fff; border: 1px solid #ddd; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
    .figure svg {{ width: 100%; height: auto; display: block; }}
    p, li {{ line-height: 1.5; }}
    code {{ background: #f0f0f0; padding: 1px 4px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Unsupervised Exploration of Rhamnose Feature Tables</h1>
      <p>This report uses the exported interpretable features only. PCA and K-means were implemented directly with <code>numpy</code> because <code>scikit-learn</code> is not installed in the current environment.</p>
    </div>
    {''.join(rows)}
  </div>
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()

    analyses = []
    specs = [
        ("raman", RAMAN_COLS, "raman_available", "Raman interpretable features"),
        ("eem", EEM_COLS, "eem_available", "Cleaned EEM interpretable features"),
        ("fusion", EEM_COLS + RAMAN_COLS, "raman_available", "Fusion interpretable features on rows with both modalities"),
    ]

    for key, cols, require_col, label in specs:
        subset_rows, X = build_matrix(rows, cols, require_col)
        if key == "fusion":
            subset_rows = [r for r in subset_rows if r.get("eem_available") == "True"]
            # rebuild to enforce both modalities
            subset_rows, X = build_matrix(subset_rows, cols, "raman_available")
        if len(subset_rows) < 3:
            continue
        X = impute_column_means(X)
        p = pca(X, n_components=2)
        labels, centers, inertia = kmeans(p["scores"], k=3, n_iter=100)
        sample_labels = [
            f"{r.get('sample_set','')}:{r.get('sample_id','')}"
            for r in subset_rows
        ]
        render_scatter_svg(
            OUT_DIR / f"{key}_pca_clusters.svg",
            f"{label}: PCA with K-means clusters",
            f"n = {len(subset_rows)} rows; cluster colors from K-means on PC scores",
            p["scores"],
            labels,
            p["explained_ratio"],
            sample_labels,
        )
        render_bar_svg(
            OUT_DIR / f"{key}_pc1_loadings.svg",
            f"{label}: strongest PC1 loadings",
            "Highest-magnitude feature contributions to the first principal component",
            top_loadings(cols, p["loadings"][0], top_n=6),
        )
        cluster_summary = summarize_clusters(subset_rows, labels)
        analyses.append(
            {
                "key": key,
                "label": label,
                "rows": subset_rows,
                "columns": cols,
                "pca": p,
                "labels": labels,
                "inertia": inertia,
                "cluster_summary": cluster_summary,
            }
        )

    # csv summary
    summary_rows = []
    for a in analyses:
        summary_rows.append(
            {
                "analysis": a["key"],
                "n_rows": len(a["rows"]),
                "n_features": len(a["columns"]),
                "pc1_explained_percent": f"{a['pca']['explained_ratio'][0]*100:.3f}",
                "pc2_explained_percent": f"{a['pca']['explained_ratio'][1]*100:.3f}",
                "kmeans_inertia": f"{a['inertia']:.3f}",
            }
        )
    with (OUT_DIR / "unsupervised_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    sections = []
    for a in analyses:
        pc1 = top_loadings(a["columns"], a["pca"]["loadings"][0], top_n=4)
        pc2 = top_loadings(a["columns"], a["pca"]["loadings"][1], top_n=4)
        cluster_lines = ["<ul>"]
        for cluster_id in sorted(a["cluster_summary"].keys()):
            item = a["cluster_summary"][cluster_id]
            top_set = item["sample_set"].most_common(2)
            top_label = [x for x in item["label"].most_common(3) if x[0]]
            cluster_lines.append(
                f"<li><strong>Cluster {cluster_id}</strong>: "
                f"sample sets {top_set}; top treatment labels {top_label}</li>"
            )
        cluster_lines.append("</ul>")
        sections.append(
            {
                "title": a["label"],
                "text": (
                    f"PC1 explains {a['pca']['explained_ratio'][0]*100:.1f}% and PC2 explains "
                    f"{a['pca']['explained_ratio'][1]*100:.1f}% of the standardized variance. "
                    f"Dominant PC1 features: {pc1}. Dominant PC2 features: {pc2}."
                ),
                "svg": "<div class='figure'>" + read_inline_svg(OUT_DIR / f"{a['key']}_pca_clusters.svg") + "</div>"
                       + "<div class='figure'>" + read_inline_svg(OUT_DIR / f"{a['key']}_pc1_loadings.svg") + "</div>",
                "extra": "".join(cluster_lines),
            }
        )
        # write score table
        score_fields = ["sample_set", "batch", "replicate", "sample_id", "matching_measurements", "legend_treatment_label", "pc1", "pc2", "cluster"]
        score_rows = []
        for row, score, cluster in zip(a["rows"], a["pca"]["scores"], a["labels"]):
            score_rows.append(
                {
                    "sample_set": row.get("sample_set", ""),
                    "batch": row.get("batch", ""),
                    "replicate": row.get("replicate", ""),
                    "sample_id": row.get("sample_id", ""),
                    "matching_measurements": row.get("matching_measurements", ""),
                    "legend_treatment_label": row.get("legend_treatment_label", ""),
                    "pc1": f"{score[0]:.6f}",
                    "pc2": f"{score[1]:.6f}",
                    "cluster": str(int(cluster)),
                }
            )
        with (OUT_DIR / f"{a['key']}_scores.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=score_fields)
            writer.writeheader()
            writer.writerows(score_rows)

    write_report(OUT_DIR / "unsupervised_report.html", sections)
    print(f"Output directory: {OUT_DIR}")
    print(f"Analyses generated: {len(analyses)}")


if __name__ == "__main__":
    main()
