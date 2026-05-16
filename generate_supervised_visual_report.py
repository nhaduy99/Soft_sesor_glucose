import csv
import html
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "supervised_monosaccharides"
BEST_MODELS_CSV = OUT_DIR / "best_models.csv"
METRICS_SUMMARY_CSV = OUT_DIR / "model_search_metrics_summary.csv"
OPTIMIZATION_CSV = OUT_DIR / "optimization_improvement_summary.csv"
PREDICTIONS_CSV = OUT_DIR / "example_predictions_seed0.csv"
TARGET_SUMMARY_CSV = OUT_DIR / "target_summary.csv"
REPORT_HTML = OUT_DIR / "comprehensive_modeling_report.html"

TARGET_LABELS = {
    "rhamnose_gL": "Rhamnose",
    "xylose_gL": "Xylose",
    "glucose_gL": "Glucose",
}

COLORS = {
    "rhamnose_gL": "#2f7d7e",
    "xylose_gL": "#7a4e9f",
    "glucose_gL": "#b5651d",
}


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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
    return f"""
    <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#fff" stroke="#d6dde5"/>
    <line x1="{x}" y1="{y+h}" x2="{x+w}" y2="{y+h}" stroke="#17202a"/>
    <line x1="{x}" y1="{y}" x2="{x}" y2="{y+h}" stroke="#17202a"/>
    <text x="{x+w/2}" y="{y+h+42}" text-anchor="middle" font-size="14">{esc(x_label)}</text>
    <text x="{x-48}" y="{y+h/2}" text-anchor="middle" font-size="14" transform="rotate(-90 {x-48} {y+h/2})">{esc(y_label)}</text>
    <text x="{x+w/2}" y="28" text-anchor="middle" font-size="18" font-weight="700">{esc(title)}</text>
    """


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
    svg = f"""<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)} bar chart">
    {axis_svg(px, py, pw, ph, "", y_label, title)}
    <line x1="{px}" y1="{zero_y:.1f}" x2="{px+pw}" y2="{zero_y:.1f}" stroke="#44546a" stroke-width="1.5"/>
    {''.join(bars)}
    </svg>"""
    return svg


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
            scatter_sections.append(f'<section class="panel">{predicted_vs_true_svg(target, rows)}</section>')
            residual_sections.append(f'<section class="panel">{residual_svg(target, rows)}</section>')
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

    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Comprehensive Monosaccharide Soft-Sensor Modelling Report</title>
  <style>
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; background: #f6f8fb; color: #17202a; line-height: 1.5; }}
    main {{ max-width: 1220px; margin: 0 auto; padding: 28px 22px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 34px 0 12px; font-size: 22px; }}
    h3 {{ margin: 0 0 10px; font-size: 17px; }}
    .muted {{ color: #5d6d7e; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .grid3 {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }}
    .panel {{ background: #fff; border: 1px solid #d6dde5; border-radius: 8px; padding: 16px; margin: 14px 0; overflow-x: auto; }}
    .callout {{ border-left: 4px solid #2d5f9a; background: #eef5ff; padding: 12px 14px; margin: 14px 0; }}
    .warn {{ border-left-color: #a85d00; background: #fff6e8; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 14px; }}
    th, td {{ border: 1px solid #d6dde5; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #eef3f8; }}
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
    <strong>Scope:</strong> supervised results are based on standards and known spikes parsed from treatment labels. Culture-sample prediction still requires quantitative HPLC monosaccharide targets.
  </section>

  <h2>Pipeline Summary</h2>
  <div class="grid3">
    <section class="panel"><h3>1. Raw inputs</h3><p>EEM CSV matrices, Raman spectra, data-description workbooks, and plate/well metadata.</p></section>
    <section class="panel"><h3>2. Inventory</h3><p>Files are joined by filename structure, experiment, plate, well, modality, and legend labels.</p></section>
    <section class="panel"><h3>3. Features</h3><p>EEM hotspots/full unfolded matrices, Raman windows/full spectra, and Raman+EEM fusion tables.</p></section>
    <section class="panel"><h3>4. Targets</h3><p>Known labels such as <code>Rha (5)</code>, <code>Xyl (0.1)</code>, <code>Glu (1)</code>, and mixes are parsed as g/L targets.</p></section>
    <section class="panel"><h3>5. Training</h3><p>Pure-NumPy Ridge, PCR, PLS1, weighted kNN, and focused kernel-ridge searches are evaluated over repeated grouped splits.</p></section>
    <section class="panel"><h3>6. Optimization</h3><p>Distance metrics, row-wise normalization, log targets, fusion features, and Laplacian/RBF kernels were compared.</p></section>
  </div>

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

  <h2>Metric Plots</h2>
  <div class="grid">
    <section class="panel">{rmse_svg}</section>
    <section class="panel">{improvement_svg}</section>
    <section class="panel">{extra_improvement_svg}</section>
  </div>

  <h2>Optimization Summary</h2>
  <section class="panel">
    <table>
      <tr><th>Target</th><th>Initial RMSE</th><th>Latest RMSE</th><th>Improvement vs initial</th><th>Extra improvement vs previous best</th><th>Met extra 20%</th></tr>
      {opt_rows}
    </table>
  </section>

  <h2>Predicted vs True Scatter Plots</h2>
  <div class="grid">{''.join(scatter_sections)}</div>

  <h2>Residual Plots</h2>
  <div class="grid">{''.join(residual_sections)}</div>

  <h2>Top Model Tables</h2>
  {''.join(top_model_sections)}

  <h2>Interpretation</h2>
  <section class="panel">
    <ul>
      <li>Rhamnose currently benefits most from Raman+EEM full fusion with a local weighted kNN model.</li>
      <li>Xylose improved slightly with a Laplacian kernel-ridge model on full EEM features and log-transformed targets.</li>
      <li>Glucose is best in this search with compact EEM interpretable features and Manhattan kNN, but grouped-split R2 remains weak.</li>
      <li>The requested extra 20% improvement beyond the previous best was not reached; current limits are likely target coverage, replicate structure, and lack of quantitative HPLC culture labels.</li>
    </ul>
  </section>

  <h2>Next Work</h2>
  <section class="panel">
    <ol>
      <li>Merge quantitative HPLC monosaccharide targets for culture samples.</li>
      <li>Add Raman baseline correction and EEM scatter-region masking before feature export.</li>
      <li>Add PARAFAC scores for interpretable EEM components.</li>
      <li>Compare this pure-NumPy search with scikit-learn PLSR/SVR and XGBoost if dependencies are available.</li>
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
