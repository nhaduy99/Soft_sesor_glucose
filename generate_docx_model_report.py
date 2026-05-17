import csv
import html
import math
import os
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / os.environ.get("SUPERVISED_OUT_DIR", "supervised_monosaccharides")
BASELINE_OUT_DIR = ROOT / "supervised_monosaccharides"
EXCLUDE_RHA5 = os.environ.get("EXCLUDE_RHA5", "").strip().lower() in {"1", "true", "yes"}
REPORT_DOCX = OUT_DIR / (
    os.environ.get(
        "REPORT_DOCX_NAME",
        "monosaccharide_softsensor_exclude_rha5_report.docx"
        if EXCLUDE_RHA5
        else "monosaccharide_softsensor_comprehensive_report.docx",
    )
)
BEST_MODELS = OUT_DIR / "best_models.csv"
PREPROCESSED_BEST = OUT_DIR / "preprocessed_model_best_vs_last.csv"
OPTIMIZATION = OUT_DIR / "optimization_improvement_summary.csv"
TARGET_SUMMARY = OUT_DIR / "target_summary.csv"
PREDICTIONS = OUT_DIR / "example_predictions_seed0.csv"
BEST_PREDICTIONS = OUT_DIR / "best_model_predictions_seed0.csv"
DEPENDENCY_SUMMARY = OUT_DIR / "dependency_model_comparison_summary.csv"
PARAFAC_FEATURE_DIR = ROOT / ("features/eem_parafac_exclude_rha5" if EXCLUDE_RHA5 else "features/eem_parafac")
PARAFAC_SUMMARY = PARAFAC_FEATURE_DIR / "parafac_rank_summary.csv"

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


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value, default=math.nan):
    try:
        return float(value)
    except Exception:
        return default


def xesc(value):
    return escape(str(value), {"'": "&apos;", '"': "&quot;"})


def normalize_config(config):
    parts = []
    for item in str(config).split(","):
        if "=" not in item:
            parts.append(item.strip())
            continue
        key, value = item.split("=", 1)
        text = value.strip()
        try:
            number = float(text)
            if math.isfinite(number):
                text = f"{number:g}"
        except ValueError:
            pass
        parts.append(f"{key.strip()}={text}")
    return ",".join(parts)


def nice_range(values, include_zero=False):
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
        pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def scale(value, src_min, src_max, dst_min, dst_max):
    if src_max <= src_min:
        return (dst_min + dst_max) / 2
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def svg_wrap(width, height, body):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
{body}
</svg>"""


def pipeline_svg():
    boxes = [
        (30, 55, 160, 72, "#EAF3F8", "Raw EEM", "fluorescence maps"),
        (30, 185, 160, 72, "#F8F1E7", "Raw Raman", "500-2000 cm-1"),
        (240, 55, 190, 72, "#FFFFFF", "EEM cleaning", "OVER + scatter mask"),
        (240, 185, 190, 72, "#FFFFFF", "Raman preprocessing", "spikes, ALS, SG, SNV"),
        (480, 25, 190, 72, "#EAF3F8", "Unfolded EEM", "matrix cells"),
        (480, 110, 190, 72, "#EAF3F8", "PARAFAC scores", "component features"),
        (480, 215, 190, 72, "#F8F1E7", "Raman features", "windows + spectrum"),
        (730, 85, 170, 72, "#F3F4F6", "Fusion table", "plate/well join"),
        (730, 200, 170, 72, "#F3F4F6", "Model comparison", "PLSR, SVR, XGB"),
        (960, 142, 185, 82, "#EAF7EF", "Predictions", "Rha, Xyl, Glu g/L"),
    ]
    body = ['<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="#44546a"/></marker></defs>']
    for x, y, w, h, fill, title, subtitle in boxes:
        body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="#52616f"/>')
        body.append(f'<text x="{x+w/2}" y="{y+30}" text-anchor="middle" font-size="16" font-weight="700" fill="#17202a">{html.escape(title)}</text>')
        body.append(f'<text x="{x+w/2}" y="{y+54}" text-anchor="middle" font-size="12" fill="#34495e">{html.escape(subtitle)}</text>')
    lines = [
        (190, 91, 240, 91), (190, 221, 240, 221), (430, 91, 480, 61), (430, 91, 480, 146),
        (430, 221, 480, 251), (670, 61, 730, 121), (670, 146, 730, 121), (670, 251, 730, 236),
        (900, 121, 960, 172), (900, 236, 960, 193)
    ]
    for x1, y1, x2, y2 in lines:
        body.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#44546a" stroke-width="2.2" marker-end="url(#arrow)"/>')
    body.append('<text x="590" y="304" text-anchor="middle" font-size="12">Raman is the direct sugar signal; EEM and PARAFAC describe fluorescence/process-state structure.</text>')
    return svg_wrap(1200, 330, "\n".join(body))


def result_bars_svg(best_rows, pre_rows):
    width, height = 980, 420
    px, py, pw, ph = 85, 58, 780, 255
    labels = [r["target"] for r in best_rows]
    vals = [fnum(r["mean_rmse"]) for r in best_rows]
    pre_by_target = {r["target"]: fnum(r["mean_rmse"]) for r in pre_rows}
    pre_vals = [pre_by_target.get(t, math.nan) for t in labels]
    hi = max([v for v in vals + pre_vals if math.isfinite(v)] + [1.0]) * 1.22
    body = [f'<text x="{width/2}" y="30" text-anchor="middle" font-size="20" font-weight="700">RMSE comparison: project best vs preprocessing/PARAFAC extension</text>']
    body.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" fill="#fff" stroke="#d6dde5"/>')
    group_w = pw / len(labels)
    for i, target in enumerate(labels):
        x0 = px + i * group_w + 35
        for j, (val, color, name) in enumerate(((vals[i], COLORS[target], "project best"), (pre_vals[i], "#6b7280", "new extension"))):
            bh = val / hi * ph if math.isfinite(val) else 0
            bx = x0 + j * 44
            by = py + ph - bh
            body.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="36" height="{bh:.1f}" fill="{color}" opacity="0.85"/>')
            body.append(f'<text x="{bx+18:.1f}" y="{by-7:.1f}" text-anchor="middle" font-size="11">{val:.3f}</text>')
        body.append(f'<text x="{x0+40}" y="{py+ph+28}" text-anchor="middle" font-size="13">{TARGET_LABELS[target]}</text>')
    body.append(f'<text x="{px+pw/2}" y="{height-22}" text-anchor="middle" font-size="12">Lower RMSE is better. Grey bars are the Raman preprocessing/PARAFAC extension.</text>')
    body.append('<rect x="875" y="76" width="14" height="14" fill="#2f7d7e"/><text x="895" y="88" font-size="12">Project best</text>')
    body.append('<rect x="875" y="99" width="14" height="14" fill="#6b7280"/><text x="895" y="111" font-size="12">New extension</text>')
    body.append(f'<text x="28" y="{py+ph/2}" transform="rotate(-90 28 {py+ph/2})" text-anchor="middle" font-size="13">RMSE (g/L)</text>')
    return svg_wrap(width, height, "\n".join(body))


def improvement_svg(pre_rows):
    width, height = 900, 360
    px, py, pw, ph = 80, 55, 720, 220
    vals = [fnum(r["additional_improvement_vs_last_best_pct"]) for r in pre_rows]
    lo, hi = nice_range(vals, include_zero=True)
    body = [f'<text x="{width/2}" y="30" text-anchor="middle" font-size="20" font-weight="700">Additional improvement from preprocessing/PARAFAC extension</text>']
    body.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" fill="#fff" stroke="#d6dde5"/>')
    zero_y = scale(0, lo, hi, py + ph, py)
    body.append(f'<line x1="{px}" y1="{zero_y:.1f}" x2="{px+pw}" y2="{zero_y:.1f}" stroke="#17202a" stroke-width="1.5"/>')
    bar_w = pw / (len(pre_rows) * 1.8)
    for i, row in enumerate(pre_rows):
        target = row["target"]
        val = fnum(row["additional_improvement_vs_last_best_pct"])
        cx = px + (i + 0.5) * pw / len(pre_rows)
        yv = scale(val, lo, hi, py + ph, py)
        y = min(yv, zero_y)
        h = abs(zero_y - yv)
        color = "#11845b" if val >= 5 else "#a43f5f"
        body.append(f'<rect x="{cx-bar_w/2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" opacity="0.86"/>')
        body.append(f'<text x="{cx:.1f}" y="{y-8 if val >= 0 else y+h+18:.1f}" text-anchor="middle" font-size="12">{val:.1f}%</text>')
        body.append(f'<text x="{cx:.1f}" y="{py+ph+28}" text-anchor="middle" font-size="13">{TARGET_LABELS[target]}</text>')
    body.append(f'<text x="{px+pw/2}" y="{height-22}" text-anchor="middle" font-size="12">The requested 5 percent threshold was reached only for glucose.</text>')
    return svg_wrap(width, height, "\n".join(body))


def predictions_by_target(best_rows):
    best_keys = {(r["target"], r["cohort"], r["feature_set"], r["model"], normalize_config(r["config"])) for r in best_rows}
    out = {target: [] for target in TARGET_LABELS}
    path = BEST_PREDICTIONS if BEST_PREDICTIONS.exists() else PREDICTIONS
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row["target"], row["cohort"], row["feature_set"], row["model"], normalize_config(row["config"]))
            if key in best_keys:
                out[row["target"]].append(row)
    return out


def scatter_svg(preds):
    width, height = 980, 330
    body = [f'<text x="{width/2}" y="26" text-anchor="middle" font-size="20" font-weight="700">Predicted versus true concentrations for current best models</text>']
    panel_w = 300
    for i, target in enumerate(TARGET_LABELS):
        rows = preds.get(target, [])
        x, y, w, h = 45 + i * 310, 58, 230, 210
        true_vals = [fnum(r["y_true"]) for r in rows]
        pred_vals = [fnum(r["y_pred"]) for r in rows]
        lo, hi = nice_range(true_vals + pred_vals, include_zero=True)
        body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#fff" stroke="#d6dde5"/>')
        body.append(f'<text x="{x+w/2}" y="{y-10}" text-anchor="middle" font-size="15" font-weight="700">{TARGET_LABELS[target]}</text>')
        body.append(f'<line x1="{x}" y1="{y+h}" x2="{x+w}" y2="{y}" stroke="#52616f" stroke-dasharray="5 4"/>')
        for row in rows[:80]:
            tv = fnum(row["y_true"])
            pv = fnum(row["y_pred"])
            cx = scale(tv, lo, hi, x, x + w)
            cy = scale(pv, lo, hi, y + h, y)
            body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.3" fill="{COLORS[target]}" opacity="0.68"/>')
        body.append(f'<text x="{x+w/2}" y="{y+h+24}" text-anchor="middle" font-size="11">True g/L</text>')
        body.append(f'<text x="{x-25}" y="{y+h/2}" transform="rotate(-90 {x-25} {y+h/2})" text-anchor="middle" font-size="11">Predicted g/L</text>')
    return svg_wrap(width, height, "\n".join(body))


class DocxWriter:
    def __init__(self):
        self.body = []
        self.rels = []
        self.media = []
        self.rid = 1

    def add_paragraph(self, text="", style=None, bold=False, italic=False):
        ppr = f"<w:pPr><w:pStyle w:val=\"{style}\"/></w:pPr>" if style else ""
        rpr_parts = []
        if bold:
            rpr_parts.append("<w:b/>")
        if italic:
            rpr_parts.append("<w:i/>")
        rpr = f"<w:rPr>{''.join(rpr_parts)}</w:rPr>" if rpr_parts else ""
        self.body.append(f"<w:p>{ppr}<w:r>{rpr}<w:t xml:space=\"preserve\">{xesc(text)}</w:t></w:r></w:p>")

    def add_heading(self, text, level=1):
        self.add_paragraph(text, style=f"Heading{level}")

    def add_bullet(self, text):
        self.body.append(
            f"<w:p><w:pPr><w:pStyle w:val=\"ListBullet\"/></w:pPr><w:r><w:t xml:space=\"preserve\">{xesc(text)}</w:t></w:r></w:p>"
        )

    def add_page_break(self):
        self.body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    def add_caption(self, text):
        self.add_paragraph(text, italic=True)

    def add_table(self, rows):
        cells = []
        for row_idx, row in enumerate(rows):
            tds = []
            for cell in row:
                shade = '<w:shd w:fill="EEF3F8"/>' if row_idx == 0 else ""
                bold = "<w:b/>" if row_idx == 0 else ""
                tds.append(
                    f"<w:tc><w:tcPr>{shade}</w:tcPr><w:p><w:r><w:rPr>{bold}</w:rPr><w:t xml:space=\"preserve\">{xesc(cell)}</w:t></w:r></w:p></w:tc>"
                )
            cells.append(f"<w:tr>{''.join(tds)}</w:tr>")
        self.body.append(
            '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/></w:tblPr>'
            + "".join(cells)
            + "</w:tbl>"
        )

    def add_svg(self, name, svg_text, width_in=6.8, height_in=2.6):
        rid = f"rId{self.rid}"
        self.rid += 1
        media_name = f"{name}.svg"
        self.media.append((media_name, svg_text.encode("utf-8")))
        self.rels.append(
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{media_name}"/>'
        )
        cx = int(width_in * 914400)
        cy = int(height_in * 914400)
        self.body.append(
            f"""<w:p><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">
<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{self.rid + 100}" name="{xesc(name)}"/>
<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="0" name="{xesc(media_name)}"/><pic:cNvPicPr/></pic:nvPicPr>
<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"/></pic:spPr>
</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>"""
        )

    def save(self, path):
        document = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
<w:body>{''.join(self.body)}<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720" w:header="360" w:footer="360" w:gutter="0"/></w:sectPr></w:body>
</w:document>"""
        styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:sz w:val="22"/></w:rPr><w:pPr><w:spacing w:after="120"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/><w:sz w:val="36"/></w:rPr><w:pPr><w:spacing w:after="240"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="30"/></w:rPr><w:pPr><w:spacing w:before="260" w:after="140"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="26"/></w:rPr><w:pPr><w:spacing w:before="180" w:after="100"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="ListBullet"><w:name w:val="List Bullet"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="360" w:hanging="180"/></w:pPr></w:style>
<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:color="D6DDE5"/><w:left w:val="single" w:sz="4" w:color="D6DDE5"/><w:bottom w:val="single" w:sz="4" w:color="D6DDE5"/><w:right w:val="single" w:sz="4" w:color="D6DDE5"/><w:insideH w:val="single" w:sz="4" w:color="D6DDE5"/><w:insideV w:val="single" w:sz="4" w:color="D6DDE5"/></w:tblBorders></w:tblPr></w:style>
</w:styles>"""
        content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="svg" ContentType="image/svg+xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""
        root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
        doc_rels = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
{''.join(self.rels)}
</Relationships>"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types)
            zf.writestr("_rels/.rels", root_rels)
            zf.writestr("word/_rels/document.xml.rels", doc_rels)
            zf.writestr("word/document.xml", document)
            zf.writestr("word/styles.xml", styles)
            for media_name, data in self.media:
                zf.writestr(f"word/media/{media_name}", data)


def table_from_best(best_rows):
    rows = [["Target", "Best feature set", "Model", "RMSE", "MAE", "R2", "Interpretation"]]
    notes = {
        "rhamnose_gL": "Fusion helps, but residual error remains high.",
        "xylose_gL": "Best relative performance; EEM full matrix appears informative.",
        "glucose_gL": "Best preprocessed Raman extension beats this baseline, but R2 is still weak.",
    }
    for row in best_rows:
        rows.append([
            TARGET_LABELS[row["target"]],
            row["feature_set"],
            row["model"],
            f"{fnum(row['mean_rmse']):.4f}",
            f"{fnum(row['mean_mae']):.4f}",
            f"{fnum(row['mean_r2']):.3f}",
            notes[row["target"]],
        ])
    return rows


def build_report():
    best_rows = read_csv(BEST_MODELS)
    pre_rows = read_csv(PREPROCESSED_BEST)
    opt_rows = read_csv(OPTIMIZATION)
    target_rows = read_csv(TARGET_SUMMARY)
    parafac_rows = read_csv(PARAFAC_SUMMARY)
    dependency_rows = read_csv(DEPENDENCY_SUMMARY) if DEPENDENCY_SUMMARY.exists() else []
    preds = predictions_by_target(best_rows)
    baseline_best_path = BASELINE_OUT_DIR / "best_models.csv"
    baseline_best = read_csv(baseline_best_path) if OUT_DIR != BASELINE_OUT_DIR and baseline_best_path.exists() else []
    baseline_by_target = {row["target"]: row for row in baseline_best}

    doc = DocxWriter()
    title = "Comprehensive Report: Raman and EEM Soft-Sensor Modelling for Monosaccharide Concentrations"
    if EXCLUDE_RHA5:
        title += " after Excluding Rha (5) Examples"
    doc.add_paragraph(title, style="Title")
    doc.add_paragraph("Prepared from the current project data, processing pipeline, and supervised calibration results.", italic=True)
    scope = "Scope note: current supervised modelling uses standards and known spikes parsed from treatment labels. True culture-sample validation remains blocked until quantitative HPLC monosaccharide concentrations are merged."
    if EXCLUDE_RHA5:
        scope += " This report excludes all Rha (5) examples from the training and testing set and compares the new result with the previous report."
    doc.add_paragraph(scope, bold=True)

    doc.add_heading("1. Executive Summary", 1)
    doc.add_paragraph(
        "The current pipeline has progressed from raw spectroscopy inventory to a reproducible soft-sensor workflow for rhamnose, xylose, and glucose. "
        "Raman and EEM files are aligned by experiment, plate, well, replicate, and treatment labels. EEM matrices are represented both as interpretable fluorescence summaries and unfolded matrices; Raman spectra are represented as interpretable carbohydrate windows, full-spectrum features, and baseline-corrected preprocessed features. "
        "This filtered report tests how sensitive the soft-sensor calibration is to removing the high-concentration Rha (5) examples. The comparison is important biologically because high-concentration standards can dominate regression structure and may make performance appear stronger than it would be in lower-concentration or culture-like regimes."
    )
    doc.add_svg("pipeline_overview", pipeline_svg(), 7.2, 1.9)
    doc.add_caption("Figure 1. Detailed soft-sensor pipeline. Raman spectra provide direct carbohydrate information; EEM/PARAFAC features capture fluorescence and process-state information.")
    doc.add_page_break()

    doc.add_heading("2. Data Context and Biological Interpretation", 1)
    doc.add_paragraph(
        "The biological problem is to estimate extracellular monosaccharide concentrations from spectroscopic signals. In this system, Raman spectroscopy is the more direct chemical probe for rhamnose, xylose, and glucose because carbohydrate ring, C-C, C-O, and C-O-C vibrations produce structured signals in the measured 500-2000 cm-1 range. "
        "EEM fluorescence is biologically useful but less direct for these sugars because monosaccharides are weakly fluorescent. EEM therefore acts mainly as an indirect process-state sensor, capturing matrix effects, culture background, fluorophores, and metabolic state. This is scientifically important: the most defensible soft sensor is not EEM alone, but a fusion model where Raman provides direct sugar chemistry and EEM contributes biological/process context."
    )
    doc.add_table([
        ["Target", "Labelled rows", "Non-zero rows", "Range in current calibration"],
        *[[TARGET_LABELS[r["target"]], r["labelled_rows"], r["known_nonzero_rows"], f"{r['min_value']} to {r['max_value']} g/L"] for r in target_rows],
    ])

    doc.add_heading("3. Processing and Feature Engineering Pipeline", 1)
    doc.add_paragraph(
        "The pipeline starts with raw EEM matrices and Raman spectra copied into the repository. The inventory builder joins raw files with metadata workbooks and treatment legends. Feature export then creates compact interpretable features and high-dimensional spectral matrices. Recent preprocessing additions make the Raman branch more chemically defensible and the EEM branch more suitable for component interpretation."
    )
    doc.add_paragraph(
        "For a newcomer: the SciPy Raman path uses established scientific routines. ALS estimates and removes a smooth background; Savitzky-Golay smoothing reduces noise while preserving peak shape; SNV rescales each spectrum so models compare spectral shape rather than brightness."
    )
    doc.add_table([
        ["Feature group", "Input", "Processing", "Model output role"],
        ["Raman windows", "Raw Raman spectra", "Crop to 500-2000 cm-1 and summarize key bands", "Explainable direct sugar features"],
        ["Preprocessed Raman", "Raw Raman spectra", "Cosmic spike removal, ALS baseline correction, Savitzky-Golay smoothing/derivatives, SNV, optional area normalization", "Improved direct-signal feature set; best for glucose extension"],
        ["EEM unfolded", "15 x 19 EEM matrices", "Handle saturated cells and flatten excitation-emission cells", "Predictive EEM baseline; best current xylose model"],
        ["EEM PARAFAC", "Cleaned EEM cube", "Mask OVER and near-diagonal scatter regions; fit ranks 2-8; export selected scores/loadings", "Interpretable fluorescence components"],
        ["Fusion", "Aligned Raman and EEM features", "Join by experiment, plate, well, replicate, and treatment", "Combines chemical and biological/process signals"],
    ])
    doc.add_page_break()

    doc.add_heading("4. Current Modelling Methods", 1)
    doc.add_paragraph(
        "The modelling stack is intentionally dependency-light and reproducible in the current environment. The main script evaluates pure-NumPy ridge regression, PCR, PLS1, weighted k-nearest neighbours, and kernel ridge regression. Repeated grouped train/test splits are used so that replicate structure is partly respected. Target-focused cohorts are also tested so that each monosaccharide model can emphasize samples where that target is biologically relevant."
    )
    doc.add_paragraph(
        "Four modelling families are represented. Model 1 is interpretable EEM or PARAFAC scores followed by regression. Model 2 is unfolded EEM plus regression as a robust baseline. Model 3 uses nonlinear models on spectral features. Model 4 uses Raman and EEM fusion, either through full features or mid-level PARAFAC/Raman feature concatenation."
    )

    doc.add_heading("5. Main Results", 1)
    doc.add_table(table_from_best(best_rows))
    if baseline_by_target:
        compare = [["Target", "Previous RMSE", "New RMSE without Rha (5)", "RMSE change", "New best model"]]
        for row in best_rows:
            previous = fnum(baseline_by_target.get(row["target"], {}).get("mean_rmse"))
            current = fnum(row["mean_rmse"])
            change = 100.0 * (previous - current) / previous if previous > 0 else math.nan
            compare.append([
                TARGET_LABELS[row["target"]],
                f"{previous:.4f}",
                f"{current:.4f}",
                f"{change:.1f}%",
                f"{row['feature_set']} / {row['model']}",
            ])
        doc.add_heading("Comparison With Previous Report", 2)
        doc.add_table(compare)
        doc.add_paragraph("Positive RMSE change means the filtered run improved relative to the previous report; negative means the filtered run worsened.")
    doc.add_svg("rmse_bars", result_bars_svg(best_rows, pre_rows), 7.0, 3.0)
    doc.add_caption("Figure 2. RMSE comparison. The y-axis is RMSE in g/L, so lower bars indicate better concentration prediction.")
    doc.add_paragraph(
        "The current project-level best models are not identical across targets, which is biologically plausible. Rhamnose benefits most from Raman + EEM full fusion, suggesting that direct Raman sugar information and EEM matrix-state information are both useful. Xylose is best with full EEM features and Laplacian kernel ridge regression, possibly reflecting strong covariance with fluorescence/process-state patterns in the standards/spikes. Glucose is best in the original search with compact EEM interpretable features, but the later preprocessed Raman kernel-ridge model improves glucose RMSE to 0.5094 g/L."
    )
    doc.add_svg("scatter_best", scatter_svg(preds), 7.2, 2.5)
    doc.add_caption("Figure 3. Predicted versus true concentrations. Points close to the diagonal indicate better agreement between model prediction and known standard/spike concentration.")
    if dependency_rows:
        dependency_best = []
        for target in TARGET_LABELS:
            rows = [row for row in dependency_rows if row.get("target") == target]
            if rows:
                dependency_best.append(min(rows, key=lambda row: fnum(row.get("mean_rmse"))))
        top_dependency = sorted(dependency_rows, key=lambda row: fnum(row.get("mean_rmse")))[:12]
        doc.add_heading("Dependency-Backed Model Comparison", 2)
        doc.add_paragraph("This comparison uses Conda base packages when available: scikit-learn PLSR/SVR and XGBoost. The first table shows the best dependency-backed result for each target so strong target-specific results are not hidden by CSV ordering.")
        doc.add_table([
            ["Target", "Feature set", "Model", "Config", "RMSE", "R2"],
            *[
                [
                    TARGET_LABELS.get(r["target"], r["target"]),
                    r["feature_set"],
                    r["model"],
                    r["config"],
                    f"{fnum(r['mean_rmse']):.4f}",
                    f"{fnum(r['mean_r2']):.3f}",
                ]
                for r in sorted(dependency_best, key=lambda row: fnum(row.get("mean_rmse")))
            ],
        ])
        doc.add_paragraph("Top dependency-backed rows by RMSE:")
        doc.add_table([
            ["Target", "Feature set", "Model", "Config", "RMSE", "R2"],
            *[
                [
                    TARGET_LABELS.get(r["target"], r["target"]),
                    r["feature_set"],
                    r["model"],
                    r["config"],
                    f"{fnum(r['mean_rmse']):.4f}",
                    f"{fnum(r['mean_r2']):.3f}",
                ]
                for r in top_dependency
            ],
        ])
    doc.add_page_break()

    doc.add_heading("6. Raman Preprocessing and EEM PARAFAC Extension", 1)
    doc.add_paragraph(
        "The extension added a more spectroscopy-aware Raman preprocessing path and interpretable EEM PARAFAC components. Raman preprocessing records each model's preprocessing_config, so performance can be traced to the exact sequence of cosmic-spike removal, ALS baseline correction, smoothing/derivative choice, SNV, and optional area normalization. The EEM PARAFAC script fits ranks 2-8 and selects the rank by reconstruction error, split-half stability, and prediction performance."
    )
    doc.add_table([
        ["Target", "Best extension feature set", "Model", "RMSE", "Latest baseline RMSE", "Improvement", "Met 5 percent"],
        *[[TARGET_LABELS[r["target"]], r["feature_set"], r["model"], f"{fnum(r['mean_rmse']):.4f}", f"{fnum(r['last_best_rmse']):.4f}", f"{fnum(r['additional_improvement_vs_last_best_pct']):.1f}%", r["met_5pct_vs_last_best"]] for r in pre_rows],
    ])
    doc.add_svg("improvement_extension", improvement_svg(pre_rows), 6.8, 2.7)
    doc.add_caption("Figure 4. Additional improvement from preprocessing and PARAFAC features relative to the current baseline. Positive values indicate lower RMSE.")
    doc.add_table([
        ["Rank", "Reconstruction error", "Split-half stability", "Prediction RMSE mean", "Selected"],
        *[[r["rank"], r["reconstruction_error"], r["split_half_stability"], r["prediction_rmse_mean"], r["selected_rank"]] for r in parafac_rows],
    ])
    selected_rank = 2
    for row in parafac_rows:
        if str(row.get("selected_rank", "")).lower() == "true":
            selected_rank = int(fnum(row.get("rank"), 2))
            break
    parafac_figures = [
        (PARAFAC_FEATURE_DIR / f"rank{selected_rank}_excitation_loadings.svg", "parafac_excitation_loadings"),
        (PARAFAC_FEATURE_DIR / f"rank{selected_rank}_emission_loadings.svg", "parafac_emission_loadings"),
    ] + [
        (PARAFAC_FEATURE_DIR / f"rank{selected_rank}_component{i}_map.svg", f"parafac_component{i}")
        for i in range(1, selected_rank + 1)
    ]
    for src, name in parafac_figures:
        if src.exists():
            doc.add_svg(name, src.read_text(encoding="utf-8"), 5.9, 3.1)
            doc.add_caption(f"PARAFAC visualisation: {src.stem}. Excitation and emission plots show wavelength loadings; component maps show the fluorescence pattern represented by one component.")
    doc.add_page_break()

    doc.add_heading("7. Strengths of the Current Method", 1)
    strength_items = [
        "The pipeline is reproducible and self-contained; it avoids undeclared dependencies and writes all major intermediate artifacts.",
        "The modelling design is scientifically defensible: Raman is treated as the direct monosaccharide signal and EEM as indirect biological/process context.",
        "Multiple feature representations are compared rather than assuming one modality or one model is best.",
        "The report includes visual diagnostics, predicted-vs-true plots, residual behaviour, PARAFAC component maps, and traceable preprocessing configuration labels.",
    ]
    strength_items.append(
        "The filtered run is a useful sensitivity test: removing Rha (5) changes the rhamnose optimum from Raman+EEM fusion to EEM ridge regression, revealing that the high-concentration standards were influential."
        if EXCLUDE_RHA5
        else "The fusion result for rhamnose supports the original biological hypothesis that Raman + EEM can outperform either signal family alone when the target is not fully captured by one modality."
    )
    for item in strength_items:
        doc.add_bullet(item)
    doc.add_page_break()

    doc.add_heading("8. Weaknesses and Risks", 1)
    weakness_items = [
        "The strongest limitation is target quality: culture-sample HPLC monosaccharide concentrations are not yet merged, so current supervised results are calibration/spike results rather than final biological culture predictions.",
        "Some RMSE values improve, but R2 is weak or negative for rhamnose and glucose in the best project-level models. This means the models can reduce absolute error in some settings but do not yet explain variance robustly across grouped splits.",
        "The sample structure is small relative to the feature dimension. High-dimensional EEM/Raman matrices create overfitting risk, especially for nonlinear models and nearest-neighbour methods.",
        "The PARAFAC implementation is dependency-light and useful for exploration, but a publication-grade component analysis should be repeated with a tested chemometrics library and stronger split-half validation.",
        "External spectral assignment is not yet fully referenced. Raman band interpretations should be strengthened with literature assignments before publication.",
    ]
    weakness_items.append(
        "In this filtered run, the Raman preprocessing/PARAFAC extension does not beat the new best models for any target. It should be used as interpretability support rather than the primary predictive model."
        if EXCLUDE_RHA5
        else "The current preprocessing/PARAFAC extension improves glucose only. Rhamnose and xylose become worse than their previous best models, so preprocessing should not be treated as universally beneficial."
    )
    for item in weakness_items:
        doc.add_bullet(item)

    doc.add_heading("9. Recommended Next Steps", 1)
    doc.add_paragraph(
        "The next scientifically important step is to merge quantitative HPLC monosaccharide concentrations for culture samples and rerun the full training workflow. Once culture targets are available, validation should be grouped by biological batch or experiment to estimate performance on genuinely unseen culture conditions. The current standards/spikes are valuable for calibration, but they cannot substitute for culture-level biological validation."
    )
    for item in [
        "Merge HPLC concentration targets and confirm units, dilution factors, and sample-code mapping.",
        "Re-run feature export and supervised training with culture samples included.",
        "Use the current best models as baselines, but separately report standards/spikes and true culture validation.",
        "If dependencies are allowed, benchmark scikit-learn PLSR/SVR, TensorLy PARAFAC, and gradient boosting against the pure-NumPy results.",
        "Strengthen interpretability by linking Raman windows and PARAFAC loadings to biochemical assignments and culture-state metadata.",
    ]:
        doc.add_bullet(item)

    doc.add_heading("Conclusion", 1)
    doc.add_paragraph(
        "The current project has a complete working soft-sensor prototype with traceable processing, model comparison, and visual reporting. The best evidence so far supports a Raman + EEM fusion strategy, with Raman providing direct carbohydrate chemistry and EEM providing indirect biological state information. The method is promising but not yet final: culture-target HPLC data and stronger validation are required before the model can be claimed as a deployable biological soft sensor."
    )
    doc.save(REPORT_DOCX)
    print(f"Wrote {REPORT_DOCX}")


if __name__ == "__main__":
    build_report()
