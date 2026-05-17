import csv
import html
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw" / "Emilie_SoftSensor"
TARGET_FEATURES_CSV = ROOT / "features" / "monosaccharide_interpretable_targets.csv"
OUT_DIR = ROOT / os.environ.get("SUPERVISED_OUT_DIR", "supervised_monosaccharides")
BASELINE_OUT_DIR = ROOT / "supervised_monosaccharides"
BEST_MODELS_CSV = OUT_DIR / "best_models.csv"
METRICS_SUMMARY_CSV = OUT_DIR / "model_search_metrics_summary.csv"
OPTIMIZATION_CSV = OUT_DIR / "optimization_improvement_summary.csv"
PREDICTIONS_CSV = OUT_DIR / "example_predictions_seed0.csv"
TARGET_SUMMARY_CSV = OUT_DIR / "target_summary.csv"
PREPROCESSED_BEST_CSV = OUT_DIR / "preprocessed_model_best_vs_last.csv"
DEPENDENCY_SUMMARY_CSV = OUT_DIR / "dependency_model_comparison_summary.csv"
PARAFAC_FEATURE_DIR = ROOT / ("features/eem_parafac_exclude_rha5" if os.environ.get("EXCLUDE_RHA5", "").strip().lower() in {"1", "true", "yes"} else "features/eem_parafac")
PARAFAC_SUMMARY_CSV = PARAFAC_FEATURE_DIR / "parafac_rank_summary.csv"
REPORT_HTML = OUT_DIR / "comprehensive_modeling_report.html"
EXCLUDE_RHA5 = os.environ.get("EXCLUDE_RHA5", "").strip().lower() in {"1", "true", "yes"}

TARGET_LABELS = {
    "rhamnose_gL": "Rhamnose",
    "xylose_gL": "Xylose",
    "glucose_gL": "Glucose",
}

COLORS = {
    "rhamnose_gL": "#0072B2",
    "xylose_gL": "#009E73",
    "glucose_gL": "#D55E00",
}
INK = "#111111"
GRID = "#E5E7EB"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find_raw_file(path_text):
    name = Path(path_text).name
    if not name:
        return None
    matches = list(RAW_DIR.rglob(name))
    return matches[0] if matches else None


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
    return xs, ys


def parse_eem(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    emission = [fnum(v) for v in rows[0][1:]]
    excitation = []
    values = []
    over = []
    for row in rows[1:]:
        excitation.append(fnum(row[0]))
        val_row, over_row = [], []
        for cell in row[1:]:
            if str(cell).strip().upper() == "OVER":
                val_row.append(math.nan)
                over_row.append(True)
            else:
                val_row.append(fnum(cell))
                over_row.append(False)
        values.append(val_row)
        over.append(over_row)
    return emission, excitation, values, over


def fnum(value, default=math.nan):
    try:
        return float(value)
    except Exception:
        return default


def esc(value):
    return html.escape(str(value))


def best_key(row):
    return (
        row["target"],
        row["cohort"],
        row["feature_set"],
        row["model"],
        row["config"],
    )


def scale(value, src_min, src_max, dst_min, dst_max):
    if not math.isfinite(value):
        return dst_min
    if src_max <= src_min:
        return (dst_min + dst_max) / 2
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def nice_bounds(values, include_zero=False):
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return 0.0, 1.0
    lo = min(clean)
    hi = max(clean)
    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    if hi == lo:
        pad = 1.0 if hi == 0 else abs(hi) * 0.1
    else:
        pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def axis_svg(x, y, w, h, x_label, y_label, title):
    ticks = []
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        tx = x + frac * w
        ty = y + frac * h
        ticks.append(f'<line x1="{tx:.1f}" y1="{y+h}" x2="{tx:.1f}" y2="{y+h+5}" stroke="{INK}" stroke-width="1"/>')
        ticks.append(f'<line x1="{x-5}" y1="{ty:.1f}" x2="{x}" y2="{ty:.1f}" stroke="{INK}" stroke-width="1"/>')
        if frac not in (0, 1):
            ticks.append(f'<line x1="{tx:.1f}" y1="{y}" x2="{tx:.1f}" y2="{y+h}" stroke="{GRID}" stroke-width="0.8"/>')
            ticks.append(f'<line x1="{x}" y1="{ty:.1f}" x2="{x+w}" y2="{ty:.1f}" stroke="{GRID}" stroke-width="0.8"/>')
    return f"""
    <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#fff"/>
    {''.join(ticks)}
    <line x1="{x}" y1="{y+h}" x2="{x+w}" y2="{y+h}" stroke="{INK}" stroke-width="1.6"/>
    <line x1="{x}" y1="{y}" x2="{x}" y2="{y+h}" stroke="{INK}" stroke-width="1.6"/>
    <text x="{x+w/2}" y="{y+h+42}" text-anchor="middle" font-size="13">{esc(x_label)}</text>
    <text x="{x-48}" y="{y+h/2}" text-anchor="middle" font-size="13" transform="rotate(-90 {x-48} {y+h/2})">{esc(y_label)}</text>
    <text x="{x+w/2}" y="28" text-anchor="middle" font-size="17" font-weight="700">{esc(title)}</text>
    """


def caption(label, text):
    return f'<p class="figcap"><strong>{esc(label)}.</strong> {esc(text)}</p>'


def pipeline_diagram_svg():
    width, height = 1180, 470
    body = ["""
    <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="#333"/></marker></defs>
    <text x="590" y="28" text-anchor="middle" font-size="18" font-weight="700">End-to-end soft-sensor modelling workflow</text>
    """]
    boxes = [
        (30, 70, 170, 82, "#EAF3F8", "Raw EEM", "Excitation-emission matrices"),
        (30, 260, 170, 82, "#F8F1E7", "Raw Raman", "Spectra, 500-2000 cm-1"),
        (245, 70, 205, 82, "#F7F7F7", "EEM cleaning", "OVER + scatter mask"),
        (245, 260, 205, 82, "#F7F7F7", "Raman preprocessing", "spikes, ALS, SG, SNV"),
        (500, 38, 205, 82, "#EAF3F8", "Unfolded EEM", "all matrix cells"),
        (500, 140, 205, 82, "#EAF3F8", "PARAFAC scores", "interpretable components"),
        (500, 260, 205, 82, "#F8F1E7", "Raman features", "windows + full spectrum"),
        (755, 105, 180, 82, "#F3F4F6", "Fusion table", "join by plate/well"),
        (755, 260, 180, 82, "#F3F4F6", "Model comparison", "PLSR, SVR, XGB, kNN"),
        (985, 175, 165, 92, "#EAF7EF", "Predictions", "Rha, Xyl, Glu g/L"),
    ]
    for x, y, w, h, fill, title, sub in boxes:
        body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" stroke="#333" stroke-width="1.1"/>')
        body.append(f'<text x="{x+w/2}" y="{y+32}" text-anchor="middle" font-size="14" font-weight="700">{esc(title)}</text>')
        body.append(f'<text x="{x+w/2}" y="{y+58}" text-anchor="middle" font-size="11">{esc(sub)}</text>')
    for x1, y1, x2, y2 in [
        (200,111,245,111), (200,301,245,301), (450,111,500,79), (450,111,500,181),
        (450,301,500,301), (705,79,755,146), (705,181,755,146), (705,301,755,301),
        (935,146,985,210), (935,301,985,232)
    ]:
        body.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#333" stroke-width="1.4" marker-end="url(#arrow)"/>')
    body.append('<text x="590" y="425" text-anchor="middle" font-size="12">Raman provides direct carbohydrate signal; EEM/PARAFAC provides indirect fluorescence and process-state information.</text>')
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Detailed modelling pipeline">{"".join(body)}</svg>'


def predicted_vs_true_svg(target, rows):
    width, height = 620, 500
    px, py, pw, ph = 82, 55, 480, 360
    y_true = [fnum(r["y_true"]) for r in rows]
    y_pred = [fnum(r["y_pred"]) for r in rows]
    lo, hi = nice_bounds(y_true + y_pred, include_zero=True)
    color = COLORS[target]
    circles = []
    for r in rows:
        xt = fnum(r["y_true"])
        yp = fnum(r["y_pred"])
        cx = scale(xt, lo, hi, px, px + pw)
        cy = scale(yp, lo, hi, py + ph, py)
        label = esc(r.get("legend_treatment_label", ""))
        circles.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="4.2" fill="{color}" opacity="0.72">'
            f"<title>{label}: true={xt:.4g}, pred={yp:.4g}</title></circle>"
        )
    diag = f'<line x1="{px}" y1="{py+ph}" x2="{px+pw}" y2="{py}" stroke="#44546a" stroke-width="2" stroke-dasharray="6 5"/>'
    ticks = []
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        val = lo + frac * (hi - lo)
        tx = scale(val, lo, hi, px, px + pw)
        ty = scale(val, lo, hi, py + ph, py)
        ticks.append(f'<line x1="{tx:.1f}" y1="{py+ph}" x2="{tx:.1f}" y2="{py+ph+6}" stroke="#17202a"/>')
        ticks.append(f'<text x="{tx:.1f}" y="{py+ph+22}" text-anchor="middle" font-size="11">{val:.2g}</text>')
        ticks.append(f'<line x1="{px-6}" y1="{ty:.1f}" x2="{px}" y2="{ty:.1f}" stroke="#17202a"/>')
        ticks.append(f'<text x="{px-10}" y="{ty+4:.1f}" text-anchor="end" font-size="11">{val:.2g}</text>')
    svg = f"""<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(target)} predicted versus true scatter plot">
    {axis_svg(px, py, pw, ph, "True concentration (g/L)", "Predicted concentration (g/L)", TARGET_LABELS[target] + " predicted vs true")}
    {diag}
    {''.join(ticks)}
    {''.join(circles)}
    </svg>"""
    return svg


def residual_svg(target, rows):
    width, height = 620, 440
    px, py, pw, ph = 82, 48, 480, 300
    y_true = [fnum(r["y_true"]) for r in rows]
    residuals = [fnum(r["residual"]) for r in rows]
    x_lo, x_hi = nice_bounds(y_true, include_zero=True)
    y_lo, y_hi = nice_bounds(residuals, include_zero=True)
    color = COLORS[target]
    circles = []
    for r in rows:
        xt = fnum(r["y_true"])
        res = fnum(r["residual"])
        cx = scale(xt, x_lo, x_hi, px, px + pw)
        cy = scale(res, y_lo, y_hi, py + ph, py)
        label = esc(r.get("legend_treatment_label", ""))
        circles.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="4" fill="{color}" opacity="0.72">'
            f"<title>{label}: residual={res:.4g}</title></circle>"
        )
    zero_y = scale(0.0, y_lo, y_hi, py + ph, py)
    zero_line = f'<line x1="{px}" y1="{zero_y:.2f}" x2="{px+pw}" y2="{zero_y:.2f}" stroke="#44546a" stroke-width="2" stroke-dasharray="5 4"/>'
    svg = f"""<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(target)} residual plot">
    {axis_svg(px, py, pw, ph, "True concentration (g/L)", "Residual: true - predicted", TARGET_LABELS[target] + " residuals")}
    {zero_line}
    {''.join(circles)}
    </svg>"""
    return svg


def bar_chart_svg(rows, value_key, title, y_label, percent=False):
    width, height = 760, 430
    px, py, pw, ph = 80, 54, 620, 270
    vals = [fnum(r[value_key]) for r in rows]
    lo = min(0.0, min(vals))
    hi = max(vals)
    if hi <= lo:
        hi = lo + 1
    bars = []
    n = len(rows)
    gap = 18
    bw = (pw - gap * (n - 1)) / max(1, n)
    zero_y = scale(0, lo, hi, py + ph, py)
    for i, row in enumerate(rows):
        target = row["target"]
        val = fnum(row[value_key])
        x = px + i * (bw + gap)
        y = scale(max(val, 0.0), lo, hi, py + ph, py)
        h = abs(zero_y - y)
        label = TARGET_LABELS.get(target, target)
        suffix = "%" if percent else ""
        bars.append(f'<rect x="{x:.1f}" y="{min(y, zero_y):.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{COLORS[target]}" opacity="0.82"/>')
        bars.append(f'<text x="{x+bw/2:.1f}" y="{py+ph+24}" text-anchor="middle" font-size="12">{esc(label)}</text>')
        bars.append(f'<text x="{x+bw/2:.1f}" y="{min(y, zero_y)-8:.1f}" text-anchor="middle" font-size="12">{val:.2f}{suffix}</text>')
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        val = lo + frac * (hi - lo)
        ty = scale(val, lo, hi, py + ph, py)
        bars.append(f'<line x1="{px-5}" y1="{ty:.1f}" x2="{px}" y2="{ty:.1f}" stroke="{INK}" stroke-width="1"/>')
        bars.append(f'<text x="{px-10}" y="{ty+4:.1f}" text-anchor="end" font-size="11">{val:.1f}{"%" if percent else ""}</text>')
    svg = f"""<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)} bar chart">
    {axis_svg(px, py, pw, ph, "", y_label, title)}
    <line x1="{px}" y1="{zero_y:.1f}" x2="{px+pw}" y2="{zero_y:.1f}" stroke="#44546a" stroke-width="1.5"/>
    {''.join(bars)}
    </svg>"""
    return svg


def select_spectroscopy_samples(rows):
    wanted = [
        ("Rha (5)", "Rhamnose 5 g/L"),
        ("Xyl (5)", "Xylose 5 g/L"),
        ("Glu (5)", "Glucose 5 g/L"),
        ("MM f/2 (1)", "Master mix 1 g/L each"),
    ]
    selected = []
    for label, display in wanted:
        candidates = [r for r in rows if r.get("legend_treatment_label") == label]
        chosen = None
        for row in candidates:
            eem_path = find_raw_file(row.get("eem_file", ""))
            raman_path = find_raw_file(row.get("raman_file", ""))
            if eem_path or raman_path:
                chosen = dict(row)
                chosen["_display_label"] = display
                chosen["_eem_path"] = eem_path
                chosen["_raman_path"] = raman_path
                break
        if chosen:
            selected.append(chosen)
    return selected


def polyline(points, color, width=2.0, opacity=0.9):
    if not points:
        return ""
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{width}" opacity="{opacity}"/>'


def raman_overlay_svg(samples):
    width, height = 860, 470
    px, py, pw, ph = 76, 48, 670, 300
    spectra = []
    for sample in samples:
        path = sample.get("_raman_path")
        if path:
            xs, ys = parse_raman(path)
            cropped = [(x, y) for x, y in zip(xs, ys) if 500 <= x <= 2000]
            if cropped:
                spectra.append((sample, cropped))
    all_x = [x for _, spec in spectra for x, _ in spec]
    all_y = [y for _, spec in spectra for _, y in spec]
    x_lo, x_hi = 500, 2000
    y_lo, y_hi = nice_bounds(all_y, include_zero=True)
    colors = ["#2f7d7e", "#7a4e9f", "#b5651d", "#2d5f9a"]
    lines, legend = [], []
    for idx, (sample, spec) in enumerate(spectra):
        color = colors[idx % len(colors)]
        pts = [(scale(x, x_lo, x_hi, px, px + pw), scale(y, y_lo, y_hi, py + ph, py)) for x, y in spec]
        lines.append(polyline(pts, color, width=2.0, opacity=0.82))
        ly = py + 22 * idx
        legend.append(f'<rect x="{px+pw+24}" y="{ly-11}" width="13" height="13" fill="{color}"/>')
        legend.append(f'<text x="{px+pw+44}" y="{ly}" font-size="12">{esc(sample["_display_label"])}</text>')
    windows = [
        (735, "C-C/C-O region"),
        (905, "sugar ring"),
        (1156, "C-O-C / glycosidic"),
        (1408, "CH bending"),
        (1523, "matrix band"),
    ]
    annotations = []
    for xval, label in windows:
        x = scale(xval, x_lo, x_hi, px, px + pw)
        annotations.append(f'<line x1="{x:.1f}" y1="{py}" x2="{x:.1f}" y2="{py+ph}" stroke="#44546a" stroke-dasharray="4 4" opacity="0.55"/>')
        annotations.append(f'<text x="{x+4:.1f}" y="{py+16}" font-size="11" fill="#44546a">{esc(label)}</text>')
    return f"""<svg viewBox="0 0 {width} {height}" role="img" aria-label="Annotated Raman spectra for monosaccharide standards">
    {axis_svg(px, py, pw, ph, "Raman shift (cm-1), cropped to useful 500-2000 range", "Intensity", "Processed Raman signal examples")}
    {''.join(annotations)}
    {''.join(lines)}
    {''.join(legend)}
    <text x="{px}" y="{py+ph+66}" font-size="13" fill="#5d6d7e">Annotated vertical guides mark candidate carbohydrate-sensitive regions used by the interpretable Raman feature windows. Raman carries the more direct monosaccharide signal, while mixtures and matrix effects change relative band intensity.</text>
    </svg>"""


def eem_heatmap_svg(sample):
    path = sample.get("_eem_path")
    if not path:
        return ""
    emission, excitation, values, over = parse_eem(path)
    width, height = 650, 500
    px, py, pw, ph = 82, 56, 440, 310
    valid = [v for row in values for v in row if math.isfinite(v)]
    lo, hi = nice_bounds(valid, include_zero=True)
    n_ex = len(excitation)
    n_em = len(emission)
    cw, ch = pw / max(1, n_em), ph / max(1, n_ex)
    cells = []
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            x = px + j * cw
            y = py + i * ch
            if over[i][j]:
                fill = "#111827"
                opacity = 0.92
            elif not math.isfinite(value) or emission[j] <= excitation[i] + 20:
                fill = "#e5e7eb"
                opacity = 1.0
            else:
                t = max(0.0, min(1.0, (value - lo) / (hi - lo))) if hi > lo else 0.0
                r = int(245 - 200 * t)
                g = int(248 - 120 * t)
                b = int(251 - 20 * t)
                fill = f"rgb({r},{g},{b})"
                opacity = 1.0
            cells.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cw+0.2:.2f}" height="{ch+0.2:.2f}" fill="{fill}" opacity="{opacity}"/>')
    ex_ticks = []
    for idx in (0, len(emission)//2, len(emission)-1):
        x = px + idx * cw + cw / 2
        ex_ticks.append(f'<text x="{x:.1f}" y="{py+ph+20}" text-anchor="middle" font-size="11">{emission[idx]:.0f}</text>')
    for idx in (0, len(excitation)//2, len(excitation)-1):
        y = py + idx * ch + ch / 2
        ex_ticks.append(f'<text x="{px-10}" y="{y+4:.1f}" text-anchor="end" font-size="11">{excitation[idx]:.0f}</text>')
    return f"""<svg viewBox="0 0 {width} {height}" role="img" aria-label="Annotated EEM heatmap for {esc(sample['_display_label'])}">
    <text x="{width/2}" y="28" text-anchor="middle" font-size="18" font-weight="700">EEM processed matrix: {esc(sample['_display_label'])}</text>
    <rect x="{px}" y="{py}" width="{pw}" height="{ph}" fill="#fff" stroke="#d6dde5"/>
    {''.join(cells)}
    <rect x="{px}" y="{py}" width="{pw}" height="{ph}" fill="none" stroke="#17202a"/>
    <text x="{px+pw/2}" y="{py+ph+42}" text-anchor="middle" font-size="13">Emission wavelength (nm)</text>
    <text x="{px-52}" y="{py+ph/2}" text-anchor="middle" font-size="13" transform="rotate(-90 {px-52} {py+ph/2})">Excitation wavelength (nm)</text>
    {''.join(ex_ticks)}
    <rect x="{px+pw+34}" y="{py+4}" width="16" height="16" fill="#111827"/><text x="{px+pw+58}" y="{py+17}" font-size="12">OVER detector saturation</text>
    <rect x="{px+pw+34}" y="{py+30}" width="16" height="16" fill="#e5e7eb"/><text x="{px+pw+58}" y="{py+43}" font-size="12">Masked scatter/invalid region</text>
    <rect x="{px+pw+34}" y="{py+56}" width="16" height="16" fill="rgb(45,128,231)"/><text x="{px+pw+58}" y="{py+69}" font-size="12">Higher fluorescence intensity</text>
    <text x="{px}" y="{py+ph+76}" font-size="13" fill="#5d6d7e">EEM is an indirect process-state signal for monosaccharides: sugars are weakly fluorescent, so useful information may come from matrix interactions, algae metabolites, or spiked-culture changes rather than a single direct sugar peak.</text>
    </svg>"""


def top_models_table(metrics_rows, target, limit=8):
    rows = [r for r in metrics_rows if r["target"] == target]
    rows = sorted(rows, key=lambda r: fnum(r["mean_rmse"]))[:limit]
    body = []
    for r in rows:
        body.append(
            "<tr>"
            f"<td>{esc(r['cohort'])}</td>"
            f"<td>{esc(r['feature_set'])}</td>"
            f"<td>{esc(r['model'])}</td>"
            f"<td>{esc(r['config'])}</td>"
            f"<td>{fnum(r['mean_rmse']):.4f}</td>"
            f"<td>{fnum(r['mean_mae']):.4f}</td>"
            f"<td>{fnum(r['mean_r2']):.3f}</td>"
            "</tr>"
        )
    return "\n".join(body)


def build_report():
    best_models = read_csv(BEST_MODELS_CSV)
    metrics_rows = read_csv(METRICS_SUMMARY_CSV)
    optimization_rows = read_csv(OPTIMIZATION_CSV)
    prediction_rows = read_csv(PREDICTIONS_CSV)
    target_summary = read_csv(TARGET_SUMMARY_CSV)
    preprocessed_best = read_csv(PREPROCESSED_BEST_CSV) if PREPROCESSED_BEST_CSV.exists() else []
    dependency_summary = read_csv(DEPENDENCY_SUMMARY_CSV) if DEPENDENCY_SUMMARY_CSV.exists() else []
    parafac_summary = read_csv(PARAFAC_SUMMARY_CSV) if PARAFAC_SUMMARY_CSV.exists() else []
    baseline_best_path = BASELINE_OUT_DIR / "best_models.csv"
    baseline_best = read_csv(baseline_best_path) if OUT_DIR != BASELINE_OUT_DIR and baseline_best_path.exists() else []
    target_feature_rows = read_csv(TARGET_FEATURES_CSV)
    spectroscopy_samples = select_spectroscopy_samples(target_feature_rows)

    best_keys = {best_key(row) for row in best_models}
    best_predictions = [row for row in prediction_rows if best_key(row) in best_keys]
    predictions_by_target = {
        target: [row for row in best_predictions if row["target"] == target]
        for target in TARGET_LABELS
    }

    scatter_sections = []
    residual_sections = []
    top_model_sections = []
    for target in TARGET_LABELS:
        rows = predictions_by_target.get(target, [])
        if rows:
            scatter_sections.append(
                f'<section class="panel">{predicted_vs_true_svg(target, rows)}'
                + caption("Predicted versus true concentrations", f"{TARGET_LABELS[target]} model predictions on the held-out grouped split. Points near the diagonal indicate better agreement.")
                + "</section>"
            )
            residual_sections.append(
                f'<section class="panel">{residual_svg(target, rows)}'
                + caption("Residual diagnostic", f"{TARGET_LABELS[target]} residuals plotted against true concentration. Random scatter around zero is preferred.")
                + "</section>"
            )
        top_model_sections.append(
            f"""
            <section class="panel">
              <h3>{TARGET_LABELS[target]}: top model configurations</h3>
              <table>
                <tr><th>Cohort</th><th>Feature set</th><th>Model</th><th>Config</th><th>RMSE</th><th>MAE</th><th>R2</th></tr>
                {top_models_table(metrics_rows, target)}
              </table>
            </section>
            """
        )

    target_rows = "\n".join(
        f"<tr><td>{esc(TARGET_LABELS.get(r['target'], r['target']))}</td><td>{esc(r['labelled_rows'])}</td>"
        f"<td>{esc(r['known_nonzero_rows'])}</td><td>{esc(r['min_value'])}</td><td>{esc(r['max_value'])}</td></tr>"
        for r in target_summary
    )

    best_rows = "\n".join(
        f"<tr><td>{esc(TARGET_LABELS.get(r['target'], r['target']))}</td><td>{esc(r['cohort'])}</td>"
        f"<td>{esc(r['feature_set'])}</td><td>{esc(r['model'])}</td><td>{esc(r['config'])}</td>"
        f"<td>{fnum(r['mean_rmse']):.4f}</td><td>{fnum(r['mean_mae']):.4f}</td>"
        f"<td>{fnum(r['mean_r2']):.3f}</td></tr>"
        for r in best_models
    )
    baseline_by_target = {row["target"]: row for row in baseline_best}
    comparison_rows = "\n".join(
        f"<tr><td>{esc(TARGET_LABELS.get(r['target'], r['target']))}</td>"
        f"<td>{fnum(baseline_by_target.get(r['target'], {}).get('mean_rmse')):.4f}</td>"
        f"<td>{fnum(r['mean_rmse']):.4f}</td>"
        f"<td>{(100.0 * (fnum(baseline_by_target.get(r['target'], {}).get('mean_rmse')) - fnum(r['mean_rmse'])) / fnum(baseline_by_target.get(r['target'], {}).get('mean_rmse'))):.1f}%</td>"
        f"<td>{esc(r['feature_set'])} / {esc(r['model'])}</td></tr>"
        for r in best_models
        if r["target"] in baseline_by_target and fnum(baseline_by_target[r["target"]].get("mean_rmse")) > 0
    )
    feature_io_rows = "\n".join(
        [
            "<tr><td>Raman interpretable windows</td><td>Raw Raman spectra cropped to 500-2000 cm-1</td><td>Integrated/summary bands near 735, 905, 1156, 1408, 1523, and 1878 cm-1 plus ratios</td><td>Direct carbohydrate-sensitive signal; compact, explainable input to Ridge/PCR/PLS/kNN/KRR models</td></tr>",
            "<tr><td>Raman full spectrum</td><td>Raw Raman spectra cropped to 500-2000 cm-1</td><td>Sampled full-spectrum intensity columns</td><td>Higher-dimensional direct sugar signal for predictive models</td></tr>",
            "<tr><td>Preprocessed Raman</td><td>Raw Raman spectra with cosmic-spike removal, asymmetric least-squares baseline correction, Savitzky-Golay smoothing or derivatives, SNV normalization, optional area normalization</td><td><code>features/raman_preprocessed_features.csv</code> with <code>rp_*</code> features and <code>preprocessing_config</code></td><td>Baseline-corrected direct sugar signal used by weighted kNN and kernel-ridge searches</td></tr>",
            "<tr><td>EEM interpretable hotspots</td><td>15 x 19 EEM matrices with saturated cells handled</td><td>Named EEM hotspot/intensity summary columns</td><td>Indirect fluorescence/process-state features; useful for compact interpretation</td></tr>",
            "<tr><td>EEM unfolded matrix</td><td>15 x 19 EEM matrices</td><td><code>eem_ex*</code> flattened excitation-emission cells</td><td>Strong EEM baseline feature set for PLS/Ridge/PCR/kNN/KRR comparisons</td></tr>",
            "<tr><td>EEM PARAFAC scores</td><td>Cleaned EEM cube with <code>OVER</code> cells and near-diagonal scatter/invalid regions masked before factorization</td><td><code>features/eem_parafac_scores.csv</code> with selected-rank component scores plus excitation/emission loadings and component maps</td><td>Interpretable latent fluorescence components, used alone and fused with Raman</td></tr>",
            "<tr><td>Raman + EEM fusion</td><td>Aligned Raman features, EEM features, and/or PARAFAC scores joined by experiment, plate, well, replicate, and treatment label</td><td>Fusion matrices for full-spectrum and mid-level models</td><td>Combines direct Raman sugar signal with indirect fluorescence/process-state signal for best robustness</td></tr>",
            "<tr><td>Model outputs</td><td>Feature matrices plus parsed concentration labels from standards/known spikes</td><td>Predicted <code>rhamnose_gL</code>, <code>xylose_gL</code>, and <code>glucose_gL</code>; RMSE, MAE, R2, residuals, and pred-vs-true plots</td><td>Soft-sensor outputs for monosaccharide concentration prediction</td></tr>",
        ]
    )

    opt_rows = "\n".join(
        f"<tr><td>{esc(TARGET_LABELS.get(r['target'], r['target']))}</td>"
        f"<td>{fnum(r['initial_baseline_rmse']):.4f}</td><td>{fnum(r['final_best_rmse']):.4f}</td>"
        f"<td>{fnum(r['rmse_improvement_pct']):.1f}%</td>"
        f"<td>{fnum(r['additional_rmse_improvement_vs_previous_best_pct']):.1f}%</td>"
        f"<td>{esc(r['met_additional_20pct_threshold'])}</td></tr>"
        for r in optimization_rows
    )

    rmse_svg = bar_chart_svg(best_models, "mean_rmse", "Best-model RMSE by target", "RMSE (g/L)")
    improvement_svg = bar_chart_svg(optimization_rows, "rmse_improvement_pct", "RMSE improvement vs initial baseline", "Improvement", percent=True)
    extra_improvement_svg = bar_chart_svg(
        optimization_rows,
        "additional_rmse_improvement_vs_previous_best_pct",
        "Additional RMSE improvement vs previous best",
        "Improvement",
        percent=True,
    )
    preprocessed_rows = "\n".join(
        f"<tr><td>{esc(TARGET_LABELS.get(r['target'], r['target']))}</td><td>{esc(r['feature_set'])}</td>"
        f"<td>{esc(r.get('preprocessing_config', ''))}</td><td>{esc(r['config'])}</td>"
        f"<td>{fnum(r['mean_rmse']):.4f}</td><td>{fnum(r['last_best_rmse']):.4f}</td>"
        f"<td>{fnum(r['additional_improvement_vs_last_best_pct']):.1f}%</td>"
        f"<td>{esc(r.get('met_5pct_vs_last_best', ''))}</td><td>{esc(r['met_10pct_vs_last_best'])}</td></tr>"
        for r in preprocessed_best
    )
    dependency_rows = "\n".join(
        f"<tr><td>{esc(TARGET_LABELS.get(r['target'], r['target']))}</td><td>{esc(r['feature_set'])}</td>"
        f"<td>{esc(r['model'])}</td><td>{esc(r['config'])}</td><td>{fnum(r['mean_rmse']):.4f}</td><td>{fnum(r['mean_r2']):.3f}</td></tr>"
        for r in sorted(dependency_summary, key=lambda row: fnum(row.get("mean_rmse")))[:20]
    )
    parafac_rows = "\n".join(
        f"<tr><td>{esc(r['rank'])}</td><td>{esc(r['reconstruction_error'])}</td>"
        f"<td>{esc(r['split_half_stability'])}</td><td>{esc(r['prediction_rmse_mean'])}</td>"
        f"<td>{esc(r['selected_rank'])}</td></tr>"
        for r in parafac_summary
    )
    selected_parafac_rank = 2
    for row in parafac_summary:
        if str(row.get("selected_rank", "")).lower() == "true":
            selected_parafac_rank = int(fnum(row.get("rank"), 2))
            break
    parafac_dir = PARAFAC_FEATURE_DIR
    parafac_svgs = []
    parafac_names = [
        f"rank{selected_parafac_rank}_excitation_loadings.svg",
        f"rank{selected_parafac_rank}_emission_loadings.svg",
    ] + [f"rank{selected_parafac_rank}_component{i}_map.svg" for i in range(1, selected_parafac_rank + 1)]
    for name in parafac_names:
        path = parafac_dir / name
        if path.exists():
            parafac_svgs.append(f'<section class="panel">{path.read_text(encoding="utf-8")}</section>')
    raman_example_svg = raman_overlay_svg(spectroscopy_samples)
    eem_example_sections = "\n".join(
        f'<section class="panel">{eem_heatmap_svg(sample)}</section>'
        for sample in spectroscopy_samples[:3]
        if sample.get("_eem_path")
    )
    interpretation_items = (
        [
            "After excluding Rha (5), rhamnose is best predicted by full EEM features with ridge regression, not by the previous Raman+EEM fusion model. This suggests the high-concentration rhamnose standards were influential in the earlier fusion result.",
            "Xylose also improves after the exclusion and is best with full EEM features plus sample-wise normalized kNN, indicating that the filtered calibration set relies strongly on matrix-level EEM structure.",
            "Glucose is essentially unchanged in the main model search, so excluding Rha (5) mainly changes rhamnose/xylose calibration rather than glucose.",
            "The Raman preprocessing/PARAFAC extension does not beat the new filtered best models, so it should be treated as interpretability support rather than the best predictive path for this filtered run.",
        ]
        if EXCLUDE_RHA5
        else [
            "Rhamnose currently benefits most from Raman+EEM full fusion with a local weighted kNN model.",
            "Xylose improved slightly with a Laplacian kernel-ridge model on full EEM features and log-transformed targets.",
            "Glucose is best in this search with compact EEM interpretable features and Manhattan kNN, but grouped-split R2 remains weak.",
            "The requested extra 20% improvement beyond the previous best was not reached; current limits are likely target coverage, replicate structure, and lack of quantitative HPLC culture labels.",
        ]
    )
    interpretation_html = "\n".join(f"<li>{esc(item)}</li>" for item in interpretation_items)

    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Comprehensive Monosaccharide Soft-Sensor Modelling Report</title>
  <style>
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; background: #f7f7f7; color: #111; line-height: 1.5; }}
    main {{ max-width: 1220px; margin: 0 auto; padding: 28px 22px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 34px 0 12px; font-size: 22px; }}
    h3 {{ margin: 0 0 10px; font-size: 17px; }}
    .muted {{ color: #5d6d7e; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .grid3 {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }}
    .panel {{ background: #fff; border: 1px solid #d9d9d9; border-radius: 4px; padding: 16px; margin: 14px 0; overflow-x: auto; }}
    .callout {{ border-left: 4px solid #333; background: #f0f4f7; padding: 12px 14px; margin: 14px 0; }}
    .warn {{ border-left-color: #777; background: #f8f8f8; }}
    .figcap {{ margin: 8px 0 0; color: #333; font-size: 13px; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 14px; }}
    th, td {{ border: 1px solid #d9d9d9; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #efefef; }}
    code {{ background: #eef3f8; border-radius: 4px; padding: 1px 4px; }}
    svg {{ width: 100%; height: auto; }}
    ol, ul {{ margin-top: 6px; }}
    @media (max-width: 900px) {{ .grid, .grid3 {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>Comprehensive Monosaccharide Soft-Sensor Modelling Report</h1>
  <p class="muted">This report summarizes the processing, feature construction, supervised training, optimization, and latest visual evaluation for rhamnose, xylose, and glucose prediction from EEM and Raman spectroscopy.</p>

  <section class="callout warn">
    <strong>Scope:</strong> supervised results are based on standards and known spikes parsed from treatment labels. Culture-sample prediction still requires quantitative HPLC monosaccharide targets. {'This version excludes all <code>Rha (5)</code> examples from training and testing.' if EXCLUDE_RHA5 else ''}
  </section>

  <h2>Pipeline Summary</h2>
  <section class="panel">
    {pipeline_diagram_svg()}
    {caption("Pipeline diagram", "Detailed flow of the current soft-sensor analysis. The filtered report excludes Rha (5) samples before model fitting; PARAFAC is fitted only on the filtered EEM cube.")}
  </section>
  <div class="grid3">
    <section class="panel"><h3>1. Raw inputs</h3><p>EEM CSV matrices, Raman spectra, data-description workbooks, and plate/well metadata.</p></section>
    <section class="panel"><h3>2. Inventory</h3><p>Files are joined by filename structure, experiment, plate, well, modality, and legend labels.</p></section>
    <section class="panel"><h3>3. Features</h3><p>EEM hotspots/full unfolded matrices, Raman windows/full spectra, and Raman+EEM fusion tables.</p></section>
    <section class="panel"><h3>4. Targets</h3><p>Known labels such as <code>Rha (5)</code>, <code>Xyl (0.1)</code>, <code>Glu (1)</code>, and mixes are parsed as g/L targets.</p></section>
    <section class="panel"><h3>5. Training</h3><p>Pure-NumPy Ridge, PCR, PLS1, weighted kNN, and focused kernel-ridge searches are evaluated over repeated grouped splits.</p></section>
    <section class="panel"><h3>6. Optimization</h3><p>Distance metrics, row-wise normalization, log targets, fusion features, and Laplacian/RBF kernels were compared.</p></section>
  </div>

  <h2>Processed Spectroscopy Signal Examples</h2>
  <section class="callout">
    <strong>Newcomer explanation:</strong> The SciPy Raman path means the same chemistry steps are now performed using tested scientific routines where available. ALS estimates a smooth background curve and subtracts it; Savitzky-Golay smooths the spectrum while preserving peak shape; SNV rescales each spectrum so models focus on shape rather than absolute brightness.
  </section>
  <section class="callout">
    <strong>How to read these images:</strong> Raman spectra are cropped to 500-2000 cm-1 because the data description says this is the useful measured range. EEM matrices show fluorescence intensity after marking saturated <code>OVER</code> cells and near-diagonal scatter/invalid regions. These examples are standards or known mixtures, so their labels provide the calibration meaning for the model.
  </section>
  <section class="panel">
    {raman_example_svg}
    {caption("Raman spectra", "Representative Raman spectra over the modelled 500-2000 cm-1 range. The x-axis is Raman shift and the y-axis is intensity; annotated windows mark carbohydrate-sensitive regions.")}
    <p><strong>Raman analysis.</strong> The overlaid standards show why Raman is the direct-signal modality for monosaccharides. The annotated regions correspond to carbohydrate-sensitive windows used in the interpretable feature export: lower-wavenumber ring/C-C/C-O regions, the 900 cm-1 sugar-ring region, and mid-range C-O-C/CH bands. Differences between pure sugars and master mix are not a single isolated peak; the model uses relative intensity patterns across windows and full-spectrum features.</p>
  </section>
  <div class="grid">
    {eem_example_sections}
  </div>
  <section class="panel">
    <h3>EEM image analysis</h3>
    <p>The EEM heatmaps show the processed 15 x 19 excitation-emission matrices used by the EEM feature sets. Dark cells are detector saturation (<code>OVER</code>) and grey cells are excluded scatter/invalid regions. Because rhamnose, xylose, and glucose are weakly fluorescent, the EEM signal is best interpreted as an indirect matrix/process-state signature rather than a direct monosaccharide peak. This explains why EEM-only models can help for xylose and glucose in the current standard/spike set, while rhamnose benefits from Raman+EEM fusion.</p>
  </section>
  <section class="callout">
    <strong>Newcomer explanation:</strong> PARAFAC is a way to separate a stack of EEM heatmaps into a small number of reusable fluorescence patterns. Each component has an excitation loading, an emission loading, and a sample score. The score is then used like a compact feature for regression.
  </section>

  <h2>Target Coverage</h2>
  <section class="panel">
    <table>
      <tr><th>Target</th><th>Labelled rows</th><th>Non-zero rows</th><th>Minimum g/L</th><th>Maximum g/L</th></tr>
      {target_rows}
    </table>
  </section>

  <h2>Best Models</h2>
  <section class="panel">
    <table>
      <tr><th>Target</th><th>Cohort</th><th>Feature set</th><th>Model</th><th>Config</th><th>RMSE</th><th>MAE</th><th>R2</th></tr>
      {best_rows}
    </table>
  </section>

  {'<h2>Comparison With Previous Report</h2><section class="panel"><table><tr><th>Target</th><th>Previous RMSE</th><th>New RMSE without Rha (5)</th><th>RMSE change</th><th>New best model</th></tr>' + comparison_rows + '</table><p>Positive RMSE change means the new filtered training set improved relative to the previous report; negative means performance worsened.</p></section>' if comparison_rows else ''}

  <h2>Feature Inputs and Model Outputs</h2>
  <section class="panel">
    <table>
      <tr><th>Feature group</th><th>Raw input</th><th>Exported modelling input</th><th>Model role / output</th></tr>
      {feature_io_rows}
    </table>
  </section>

  <h2>Metric Plots</h2>
  <div class="grid">
    <section class="panel">{rmse_svg}{caption("Best-model RMSE", "Root mean squared error in g/L for each target. Lower values indicate smaller prediction error.")}</section>
    <section class="panel">{improvement_svg}{caption("Improvement versus baseline", "Percentage RMSE reduction relative to the initial baseline model.")}</section>
    <section class="panel">{extra_improvement_svg}{caption("Additional improvement", "Additional percentage RMSE reduction relative to the previous best model.")}</section>
  </div>

  <h2>Optimization Summary</h2>
  <section class="panel">
    <table>
      <tr><th>Target</th><th>Initial RMSE</th><th>Latest RMSE</th><th>Improvement vs initial</th><th>Extra improvement vs previous best</th><th>Met extra 20%</th></tr>
      {opt_rows}
    </table>
  </section>

  <h2>Raman Preprocessing and EEM PARAFAC Extensions</h2>
  <section class="panel">
    <p>The latest extension adds Raman preprocessing configurations with cosmic-spike removal, asymmetric least-squares baseline correction, Savitzky-Golay smoothing or derivatives, SNV normalization, and optional area normalization. Every result row stores a <code>preprocessing_config</code>. EEM PARAFAC scores were exported after rank selection using reconstruction error, split-half stability, and prediction performance.</p>
    <p>Plain-language interpretation: Raman preprocessing cleans the direct sugar spectrum before modelling. EEM PARAFAC compresses each fluorescence heatmap into component scores so the model can use interpretable fluorescence patterns instead of hundreds of raw cells.</p>
    <table>
      <tr><th>Target</th><th>Best new feature set</th><th>Preprocessing config</th><th>Model config</th><th>New RMSE</th><th>Last best RMSE</th><th>Improvement</th><th>Met 5%</th><th>Met 10%</th></tr>
      {preprocessed_rows}
    </table>
  </section>
  <section class="panel">
    <h3>PARAFAC rank selection</h3>
    <table>
      <tr><th>Rank</th><th>Reconstruction error</th><th>Split-half stability</th><th>Prediction RMSE mean</th><th>Selected</th></tr>
      {parafac_rows}
    </table>
    <p>The selected rank is the number of fluorescence components retained. With TensorLy available in Conda base, the filtered run selected rank 6. PARAFAC is primarily used here for interpretation; it did not beat the best filtered predictive models.</p>
  </section>
  <div class="grid">
    {''.join(parafac_svgs)}
  </div>

  {'<h2>Dependency-Backed Model Comparison</h2><section class="panel"><p>This section uses Conda base packages when installed: scikit-learn PLSR/SVR and XGBoost.</p><table><tr><th>Target</th><th>Feature set</th><th>Model</th><th>Config</th><th>RMSE</th><th>R2</th></tr>' + dependency_rows + '</table></section>' if dependency_rows else ''}

  <h2>Predicted vs True Scatter Plots</h2>
  <div class="grid">{''.join(scatter_sections)}</div>

  <h2>Residual Plots</h2>
  <div class="grid">{''.join(residual_sections)}</div>

  <h2>Top Model Tables</h2>
  {''.join(top_model_sections)}

  <h2>Interpretation</h2>
  <section class="panel">
    <ul>
      {interpretation_html}
    </ul>
  </section>

  <h2>Next Work</h2>
  <section class="panel">
    <ol>
      <li>Merge quantitative HPLC monosaccharide targets for culture samples.</li>
      <li>Merge quantitative HPLC culture-target concentrations; this is still the main blocker for biological validation.</li>
      <li>Review dependency-backed PLSR/SVR/XGBoost results target by target before replacing the current filtered baselines.</li>
    </ol>
  </section>
</main>
</body>
</html>
"""
    REPORT_HTML.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_HTML}")
    print(f"Best prediction rows plotted: {len(best_predictions)}")


if __name__ == "__main__":
    build_report()
