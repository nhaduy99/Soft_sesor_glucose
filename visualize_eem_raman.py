import csv
import math
from pathlib import Path


ROOT = Path(
    r"C:\Users\nhadu\OneDrive - UTS\C3_UTS\Soft_Sensor_Loc\Emilie data\Raw_data\Emilie_SoftSensor"
)
DATA_ROOT = ROOT / "Emilie_SoftSensor"
OUT_DIR = ROOT / "Codex_inventory" / "visualizations"


def mean(values):
    return sum(values) / len(values) if values else 0.0


def stdev(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def load_eem_files():
    files = []
    for sub in ("Raw_data_Plate/EEM", "Raw_data_Flask/EEM"):
        files.extend(sorted((DATA_ROOT / sub).glob("*.csv")))
    return files


def load_raman_files():
    files = []
    for sub in ("Raw_data_Plate/Raman", "Raw_data_Flask/Raman"):
        files.extend(sorted((DATA_ROOT / sub).glob("*.CSV")))
        files.extend(sorted((DATA_ROOT / sub).glob("*.csv")))
    unique = {}
    for path in files:
        unique[str(path).lower()] = path
    return list(unique.values())


def aggregate_eem(files):
    emission = None
    excitation = None
    matrix_sum = []
    matrix_count = []
    over_count = []

    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        if emission is None:
            emission = [float(v) for v in rows[0][1:]]
            excitation = [float(r[0]) for r in rows[1:]]
            n_rows = len(excitation)
            n_cols = len(emission)
            matrix_sum = [[0.0] * n_cols for _ in range(n_rows)]
            matrix_count = [[0] * n_cols for _ in range(n_rows)]
            over_count = [[0] * n_cols for _ in range(n_rows)]

        for i, row in enumerate(rows[1:]):
            for j, cell in enumerate(row[1:]):
                if cell == "OVER":
                    over_count[i][j] += 1
                    continue
                value = float(cell)
                matrix_sum[i][j] += value
                matrix_count[i][j] += 1

    mean_matrix = []
    over_fraction = []
    for i in range(len(matrix_sum)):
        mean_row = []
        over_row = []
        for j in range(len(matrix_sum[0])):
            cnt = matrix_count[i][j]
            mean_row.append(matrix_sum[i][j] / cnt if cnt else 0.0)
            over_row.append(over_count[i][j] / len(files) if files else 0.0)
        mean_matrix.append(mean_row)
        over_fraction.append(over_row)

    return {
        "files": len(files),
        "excitation": excitation,
        "emission": emission,
        "mean_matrix": mean_matrix,
        "over_fraction": over_fraction,
    }


def aggregate_raman(files, min_shift=500.0, max_shift=2000.0):
    x_axis = None
    spectra = []
    for path in files:
        shifts = []
        intensities = []
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                shift = float(row[0])
                intensity = float(row[1])
                if min_shift <= shift <= max_shift:
                    shifts.append(shift)
                    intensities.append(intensity)
        if x_axis is None:
            x_axis = shifts
        spectra.append(intensities)

    mean_spec = []
    std_spec = []
    for idx in range(len(x_axis)):
        vals = [spec[idx] for spec in spectra]
        mean_spec.append(mean(vals))
        std_spec.append(stdev(vals))

    return {
        "files": len(files),
        "shift": x_axis,
        "spectra": spectra,
        "mean": mean_spec,
        "std": std_spec,
    }


def local_maxima_1d(x, y, min_spacing=80.0, top_n=6):
    candidates = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            prominence = y[i] - 0.5 * (y[i - 1] + y[i + 1])
            candidates.append((prominence, x[i], y[i], i))
    candidates.sort(reverse=True)

    selected = []
    for _, xi, yi, idx in candidates:
        if all(abs(xi - sx) >= min_spacing for sx, _, _ in selected):
            selected.append((xi, yi, idx))
        if len(selected) >= top_n:
            break
    selected.sort(key=lambda item: item[0])
    return selected


def hotspot_maxima_2d(exc, em, mat, top_n=6):
    n_rows = len(mat)
    n_cols = len(mat[0])
    candidates = []
    for i in range(n_rows):
        for j in range(n_cols):
            value = mat[i][j]
            neighbors = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ii = i + di
                    jj = j + dj
                    if 0 <= ii < n_rows and 0 <= jj < n_cols:
                        neighbors.append(mat[ii][jj])
            if all(value >= n for n in neighbors):
                candidates.append((value, exc[i], em[j], i, j))
    candidates.sort(reverse=True)

    selected = []
    for value, ex, emi, i, j in candidates:
        if all(abs(ex - sx) >= 40 or abs(emi - sy) >= 40 for _, sx, sy, _, _ in selected):
            selected.append((value, ex, emi, i, j))
        if len(selected) >= top_n:
            break
    selected.sort(key=lambda item: item[0], reverse=True)
    return selected


def cleaned_eem_matrix(exc, em, mean_mat, over_fraction, over_threshold=0.05, scatter_margin=20.0):
    cleaned = []
    for i, ex in enumerate(exc):
        row = []
        for j, emi in enumerate(em):
            value = mean_mat[i][j]
            if over_fraction[i][j] > over_threshold:
                row.append(None)
            elif emi <= ex + scatter_margin:
                row.append(None)
            else:
                row.append(value)
        cleaned.append(row)
    return cleaned


def hotspot_maxima_2d_masked(exc, em, mat, top_n=6):
    n_rows = len(mat)
    n_cols = len(mat[0])
    candidates = []
    for i in range(n_rows):
        for j in range(n_cols):
            value = mat[i][j]
            if value is None:
                continue
            neighbors = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ii = i + di
                    jj = j + dj
                    if 0 <= ii < n_rows and 0 <= jj < n_cols and mat[ii][jj] is not None:
                        neighbors.append(mat[ii][jj])
            if neighbors and all(value >= n for n in neighbors):
                candidates.append((value, exc[i], em[j], i, j))
    candidates.sort(reverse=True)

    selected = []
    for value, ex, emi, i, j in candidates:
        if all(abs(ex - sx) >= 40 or abs(emi - sy) >= 40 for _, sx, sy, _, _ in selected):
            selected.append((value, ex, emi, i, j))
        if len(selected) >= top_n:
            break
    selected.sort(key=lambda item: item[0], reverse=True)
    return selected


def color_map(value, vmin, vmax):
    if vmax <= vmin:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    r = int(25 + 230 * t)
    g = int(40 + 160 * (1.0 - abs(t - 0.5) * 2))
    b = int(210 - 170 * t)
    return f"rgb({r},{g},{b})"


def svg_header(width, height):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: Arial, sans-serif; fill: #111; }',
        '.title { font-size: 24px; font-weight: bold; }',
        '.subtitle { font-size: 13px; fill: #444; }',
        '.axis { stroke: #222; stroke-width: 1.2; }',
        '.grid { stroke: #ddd; stroke-width: 1; }',
        '.small { font-size: 11px; }',
        '.label { font-size: 12px; }',
        '.anno { font-size: 12px; font-weight: bold; fill: #7a1010; }',
        '</style>',
    ]


def write_svg(path, lines):
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def line_path(xs, ys, x0, y0, w, h, xmin, xmax, ymin, ymax):
    coords = []
    for x, y in zip(xs, ys):
        px = x0 + (x - xmin) / (xmax - xmin) * w
        py = y0 + h - (y - ymin) / (ymax - ymin) * h
        coords.append((px, py))
    if not coords:
        return ""
    return "M " + " L ".join(f"{px:.2f} {py:.2f}" for px, py in coords)


def polygon_band(xs, lower, upper, x0, y0, w, h, xmin, xmax, ymin, ymax):
    top = []
    bottom = []
    for x, y in zip(xs, upper):
        px = x0 + (x - xmin) / (xmax - xmin) * w
        py = y0 + h - (y - ymin) / (ymax - ymin) * h
        top.append((px, py))
    for x, y in reversed(list(zip(xs, lower))):
        px = x0 + (x - xmin) / (xmax - xmin) * w
        py = y0 + h - (y - ymin) / (ymax - ymin) * h
        bottom.append((px, py))
    pts = top + bottom
    return " ".join(f"{px:.2f},{py:.2f}" for px, py in pts)


def render_raman_mean_svg(data, peaks, out_path):
    width, height = 1200, 760
    x0, y0, w, h = 90, 120, 900, 500
    xs = data["shift"]
    mean_y = data["mean"]
    std_y = data["std"]
    lower = [max(0.0, m - s) for m, s in zip(mean_y, std_y)]
    upper = [m + s for m, s in zip(mean_y, std_y)]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(lower), max(upper)

    lines = svg_header(width, height)
    lines += [
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        '<text x="90" y="55" class="title">All Raman Spectra: Mean Signature with Peak Annotations</text>',
        f'<text x="90" y="82" class="subtitle">n = {data["files"]} locally available Raman files; mean spectrum shown with ±1 SD envelope, cropped to 500-2000 cm-1</text>',
    ]
    for frac in range(0, 6):
        gy = y0 + h - frac * h / 5
        gv = ymin + frac * (ymax - ymin) / 5
        lines.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x0+w}" y2="{gy:.1f}" class="grid"/>')
        lines.append(f'<text x="{x0-10}" y="{gy+4:.1f}" text-anchor="end" class="small">{gv:.0f}</text>')
    for frac in range(0, 7):
        gx = x0 + frac * w / 6
        gv = xmin + frac * (xmax - xmin) / 6
        lines.append(f'<line x1="{gx:.1f}" y1="{y0}" x2="{gx:.1f}" y2="{y0+h}" class="grid"/>')
        lines.append(f'<text x="{gx:.1f}" y="{y0+h+22}" text-anchor="middle" class="small">{gv:.0f}</text>')
    lines += [
        f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" class="axis"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}" class="axis"/>',
        f'<text x="{x0+w/2:.1f}" y="{y0+h+50}" text-anchor="middle" class="label">Raman shift (cm-1)</text>',
        f'<text x="28" y="{y0+h/2:.1f}" transform="rotate(-90 28,{y0+h/2:.1f})" text-anchor="middle" class="label">Intensity (a.u.)</text>',
    ]
    band = polygon_band(xs, lower, upper, x0, y0, w, h, xmin, xmax, ymin, ymax)
    path = line_path(xs, mean_y, x0, y0, w, h, xmin, xmax, ymin, ymax)
    lines.append(f'<polygon points="{band}" fill="rgba(123,171,223,0.35)" style="fill:#cfe3f7; stroke:none; opacity:0.75"/>')
    lines.append(f'<path d="{path}" fill="none" stroke="#1f4e79" stroke-width="2.5"/>')

    for idx, (px, py, _) in enumerate(peaks):
        sx = x0 + (px - xmin) / (xmax - xmin) * w
        sy = y0 + h - (py - ymin) / (ymax - ymin) * h
        text_y = y0 + 40 + (idx % 2) * 22
        text_x = 1010
        lines.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="4.5" fill="#b22222"/>')
        lines.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{text_x-12}" y2="{text_y-5}" stroke="#b22222" stroke-width="1.4"/>')
        lines.append(f'<text x="{text_x}" y="{text_y}" class="anno">Peak at {px:.0f} cm-1</text>')
        lines.append(f'<text x="{text_x}" y="{text_y+14}" class="small">mean intensity {py:.0f}</text>')

    lines.append('<text x="90" y="700" class="subtitle">Interpretation note: annotated peaks are dominant maxima in the mean Raman spectrum and may correspond to strong biochemical signatures; exact assignments require domain-specific validation.</text>')
    write_svg(out_path, lines)


def bin_raman_heatmap(data, n_bins=220):
    spectra = data["spectra"]
    bin_size = max(1, len(data["shift"]) // n_bins)
    binned = []
    for spec in spectra:
        row = []
        for start in range(0, len(spec), bin_size):
            chunk = spec[start:start + bin_size]
            row.append(mean(chunk))
        binned.append(row)
    return binned, bin_size


def render_raman_heatmap_svg(data, out_path):
    heat, bin_size = bin_raman_heatmap(data)
    width, height = 1200, 800
    x0, y0, w, h = 90, 110, 950, 560
    rows = len(heat)
    cols = len(heat[0])
    flat = [v for row in heat for v in row]
    vmin, vmax = min(flat), max(flat)

    lines = svg_header(width, height)
    lines += [
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        '<text x="90" y="55" class="title">All Raman Spectra: Sample-by-Sample Heatmap</text>',
        f'<text x="90" y="82" class="subtitle">n = {rows} spectra, downsampled along Raman shift for readability; warmer colors indicate stronger intensity</text>',
    ]
    cw = w / cols
    ch = h / rows
    for i, row in enumerate(heat):
        for j, value in enumerate(row):
            color = color_map(value, vmin, vmax)
            x = x0 + j * cw
            y = y0 + i * ch
            lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cw+0.2:.2f}" height="{ch+0.2:.2f}" fill="{color}" stroke="none"/>')

    lines += [
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" class="axis"/>',
        f'<text x="{x0+w/2:.1f}" y="{y0+h+40}" text-anchor="middle" class="label">Raman shift bins across 500-2000 cm-1</text>',
        f'<text x="30" y="{y0+h/2:.1f}" transform="rotate(-90 30,{y0+h/2:.1f})" text-anchor="middle" class="label">Samples</text>',
    ]
    for frac in range(0, 6):
        gx = x0 + frac * w / 5
        start_idx = int(frac * (len(data["shift"]) - 1) / 5)
        label = data["shift"][start_idx]
        lines.append(f'<text x="{gx:.1f}" y="{y0+h+20}" text-anchor="middle" class="small">{label:.0f}</text>')
    for frac in range(0, 5):
        gy = y0 + frac * h / 4
        label = int(frac * rows / 4)
        lines.append(f'<text x="{x0-10}" y="{gy+4:.1f}" text-anchor="end" class="small">{label}</text>')

    lines.append('<text x="90" y="715" class="subtitle">Use this panel to inspect heterogeneity: horizontal bands indicate sample-level intensity differences; vertical bands indicate consistently strong Raman regions.</text>')
    write_svg(out_path, lines)


def render_eem_heatmap_svg(data, peaks, out_path):
    exc = data["excitation"]
    em = data["emission"]
    mat = data["mean_matrix"]
    flat = [v for row in mat for v in row]
    vmin, vmax = min(flat), max(flat)
    width, height = 1200, 780
    x0, y0, w, h = 120, 110, 720, 520
    rows = len(exc)
    cols = len(em)
    cw = w / cols
    ch = h / rows

    lines = svg_header(width, height)
    lines += [
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        '<text x="120" y="55" class="title">All EEM Data: Mean Fluorescence Map with Hotspot Annotations</text>',
        f'<text x="120" y="82" class="subtitle">n = {data["files"]} locally available EEM files; each cell represents mean fluorescence intensity at one excitation-emission pair</text>',
    ]
    for i, ex in enumerate(exc):
        for j, emi in enumerate(em):
            val = mat[i][j]
            color = color_map(val, vmin, vmax)
            x = x0 + j * cw
            y = y0 + i * ch
            lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cw+0.2:.2f}" height="{ch+0.2:.2f}" fill="{color}" stroke="none"/>')
    lines.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" class="axis"/>')

    for i, ex in enumerate(exc):
        y = y0 + i * ch + ch * 0.65
        lines.append(f'<text x="{x0-12}" y="{y:.1f}" text-anchor="end" class="small">{int(ex)}</text>')
    for j, emi in enumerate(em):
        x = x0 + j * cw + cw / 2
        lines.append(f'<text x="{x:.1f}" y="{y0+h+18}" text-anchor="middle" class="small">{int(emi)}</text>')

    lines += [
        f'<text x="{x0+w/2:.1f}" y="{y0+h+48}" text-anchor="middle" class="label">Emission wavelength (nm)</text>',
        f'<text x="42" y="{y0+h/2:.1f}" transform="rotate(-90 42,{y0+h/2:.1f})" text-anchor="middle" class="label">Excitation wavelength (nm)</text>',
    ]
    legend_x = 900
    legend_y = 140
    for idx, (value, ex, emi, i, j) in enumerate(peaks):
        cx = x0 + j * cw + cw / 2
        cy = y0 + i * ch + ch / 2
        text_y = legend_y + idx * 58
        lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="none" stroke="#8b0000" stroke-width="2"/>')
        lines.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{legend_x-10}" y2="{text_y-4}" stroke="#8b0000" stroke-width="1.4"/>')
        lines.append(f'<text x="{legend_x}" y="{text_y}" class="anno">Hotspot {idx+1}: Ex {int(ex)} / Em {int(emi)} nm</text>')
        lines.append(f'<text x="{legend_x}" y="{text_y+14}" class="small">mean intensity {value:.0f}</text>')
    lines.append('<text x="120" y="700" class="subtitle">Annotated hotspots are dominant local maxima in the mean EEM landscape. They highlight wavelength pairs that consistently carry strong fluorescence signal across the dataset.</text>')
    write_svg(out_path, lines)


def render_eem_over_svg(data, out_path):
    exc = data["excitation"]
    em = data["emission"]
    mat = data["over_fraction"]
    flat = [v for row in mat for v in row]
    vmin, vmax = min(flat), max(flat)
    width, height = 1200, 780
    x0, y0, w, h = 120, 110, 720, 520
    rows = len(exc)
    cols = len(em)
    cw = w / cols
    ch = h / rows

    max_i = max(range(rows), key=lambda i: max(mat[i]))
    max_j = max(range(cols), key=lambda j: max(mat[i][j] for i in range(rows)))
    max_val = max(flat)

    lines = svg_header(width, height)
    lines += [
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        '<text x="120" y="55" class="title">EEM Detector Saturation Map: Fraction of OVER Values</text>',
        f'<text x="120" y="82" class="subtitle">Computed across {data["files"]} EEM files; cells near 1.0 frequently exceeded detector linear range</text>',
    ]
    for i, ex in enumerate(exc):
        for j, emi in enumerate(em):
            val = mat[i][j]
            color = color_map(val, vmin, vmax if vmax > 0 else 1.0)
            x = x0 + j * cw
            y = y0 + i * ch
            lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cw+0.2:.2f}" height="{ch+0.2:.2f}" fill="{color}" stroke="none"/>')
    lines.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" class="axis"/>')
    for i, ex in enumerate(exc):
        y = y0 + i * ch + ch * 0.65
        lines.append(f'<text x="{x0-12}" y="{y:.1f}" text-anchor="end" class="small">{int(ex)}</text>')
    for j, emi in enumerate(em):
        x = x0 + j * cw + cw / 2
        lines.append(f'<text x="{x:.1f}" y="{y0+h+18}" text-anchor="middle" class="small">{int(emi)}</text>')
    lines += [
        f'<text x="{x0+w/2:.1f}" y="{y0+h+48}" text-anchor="middle" class="label">Emission wavelength (nm)</text>',
        f'<text x="42" y="{y0+h/2:.1f}" transform="rotate(-90 42,{y0+h/2:.1f})" text-anchor="middle" class="label">Excitation wavelength (nm)</text>',
    ]
    cx = x0 + max_j * cw + cw / 2
    cy = y0 + max_i * ch + ch / 2
    lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="none" stroke="#8b0000" stroke-width="2.2"/>')
    lines.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="930" y2="180" stroke="#8b0000" stroke-width="1.5"/>')
    lines.append(f'<text x="940" y="180" class="anno">Highest saturation: Ex {int(exc[max_i])} / Em {int(em[max_j])} nm</text>')
    lines.append(f'<text x="940" y="196" class="small">OVER fraction = {max_val:.2f}</text>')
    lines.append('<text x="120" y="700" class="subtitle">This panel identifies wavelength pairs where the detector frequently saturated. Those regions should be treated cautiously during modeling, masking, or imputation.</text>')
    write_svg(out_path, lines)


def render_eem_cleaned_svg(data, cleaned_mat, peaks, out_path):
    exc = data["excitation"]
    em = data["emission"]
    valid = [v for row in cleaned_mat for v in row if v is not None]
    vmin, vmax = (min(valid), max(valid)) if valid else (0.0, 1.0)
    width, height = 1200, 780
    x0, y0, w, h = 120, 110, 720, 520
    rows = len(exc)
    cols = len(em)
    cw = w / cols
    ch = h / rows

    lines = svg_header(width, height)
    lines += [
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        '<text x="120" y="55" class="title">Cleaned EEM Mean Map: Saturation- and Scatter-Masked Fluorescence</text>',
        '<text x="120" y="82" class="subtitle">Cells with >5% OVER fraction or emission <= excitation + 20 nm are masked before hotspot detection</text>',
    ]
    for i, ex in enumerate(exc):
        for j, emi in enumerate(em):
            val = cleaned_mat[i][j]
            x = x0 + j * cw
            y = y0 + i * ch
            if val is None:
                lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cw+0.2:.2f}" height="{ch+0.2:.2f}" fill="#f2f2f2" stroke="none"/>')
            else:
                color = color_map(val, vmin, vmax)
                lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cw+0.2:.2f}" height="{ch+0.2:.2f}" fill="{color}" stroke="none"/>')
    lines.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" class="axis"/>')
    for i, ex in enumerate(exc):
        y = y0 + i * ch + ch * 0.65
        lines.append(f'<text x="{x0-12}" y="{y:.1f}" text-anchor="end" class="small">{int(ex)}</text>')
    for j, emi in enumerate(em):
        x = x0 + j * cw + cw / 2
        lines.append(f'<text x="{x:.1f}" y="{y0+h+18}" text-anchor="middle" class="small">{int(emi)}</text>')
    lines += [
        f'<text x="{x0+w/2:.1f}" y="{y0+h+48}" text-anchor="middle" class="label">Emission wavelength (nm)</text>',
        f'<text x="42" y="{y0+h/2:.1f}" transform="rotate(-90 42,{y0+h/2:.1f})" text-anchor="middle" class="label">Excitation wavelength (nm)</text>',
    ]
    legend_x = 900
    legend_y = 150
    for idx, (value, ex, emi, i, j) in enumerate(peaks):
        cx = x0 + j * cw + cw / 2
        cy = y0 + i * ch + ch / 2
        text_y = legend_y + idx * 58
        lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="none" stroke="#8b0000" stroke-width="2"/>')
        lines.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{legend_x-10}" y2="{text_y-4}" stroke="#8b0000" stroke-width="1.4"/>')
        lines.append(f'<text x="{legend_x}" y="{text_y}" class="anno">Clean hotspot {idx+1}: Ex {int(ex)} / Em {int(emi)} nm</text>')
        lines.append(f'<text x="{legend_x}" y="{text_y+14}" class="small">mean intensity {value:.0f}</text>')
    lines.append('<text x="120" y="700" class="subtitle">This cleaned panel is the more defensible fluorescence view for downstream ML feature selection because obvious saturation and near-diagonal scatter are suppressed.</text>')
    write_svg(out_path, lines)


def read_inline_svg(path):
    text = path.read_text(encoding="utf-8")
    start = text.find("<svg")
    return text[start:] if start >= 0 else text


def write_html_report(eem_data, raman_data, raman_peaks, eem_raw_peaks, eem_clean_peaks, out_path):
    vis_dir = out_path.parent
    raman_svg = read_inline_svg(vis_dir / "raman_mean_annotated.svg")
    eem_svg = read_inline_svg(vis_dir / "eem_mean_annotated.svg")
    eem_clean_svg = read_inline_svg(vis_dir / "eem_cleaned_annotated.svg")
    over_svg = read_inline_svg(vis_dir / "eem_over_fraction.svg")

    peak_items = "".join(
        f"<li><strong>{px:.0f} cm<sup>-1</sup></strong>: mean intensity {py:.0f}</li>"
        for px, py, _ in raman_peaks
    )
    raw_hotspots = "".join(
        f"<li><strong>Ex {int(ex)} / Em {int(em)} nm</strong>: mean intensity {value:.0f}</li>"
        for value, ex, em, _, _ in eem_raw_peaks
    )
    clean_hotspots = "".join(
        f"<li><strong>Ex {int(ex)} / Em {int(em)} nm</strong>: mean intensity {value:.0f}</li>"
        for value, ex, em, _, _ in eem_clean_peaks
    )
    max_over = max(v for row in eem_data["over_fraction"] for v in row)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Raman and EEM Visual Analysis Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f7f7f7; color: #111; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 32px; }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    p, li {{ line-height: 1.5; }}
    .card {{ background: white; border: 1px solid #ddd; padding: 24px; margin: 0 0 24px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
    .meta {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 16px 0 0 0; }}
    .meta div {{ background: #fafafa; border: 1px solid #e3e3e3; padding: 12px; }}
    .figure svg {{ width: 100%; height: auto; display: block; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
    .note {{ color: #444; }}
    code {{ background: #f0f0f0; padding: 1px 4px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Raman and EEM Visual Analysis Report</h1>
      <p class="note">Prepared from all locally available raw files in <code>Raw_data_Plate</code> and <code>Raw_data_Flask</code>. Duplicate flat exports in <code>Raw_data_files_ALL</code> were intentionally excluded.</p>
      <div class="meta">
        <div><strong>Raman files analysed</strong><br>{raman_data["files"]}</div>
        <div><strong>EEM files analysed</strong><br>{eem_data["files"]}</div>
        <div><strong>Maximum EEM OVER fraction</strong><br>{max_over:.2f}</div>
      </div>
    </div>

    <div class="card">
      <h2>Interpretive Summary</h2>
      <p>The Raman mean spectrum shows a limited number of consistently strong bands across the dataset. The most prominent regions are near <strong>1156 cm<sup>-1</sup></strong> and <strong>1523 cm<sup>-1</sup></strong>, which are reasonable candidate features for downstream regression because they remain visible after averaging across hundreds of spectra.</p>
      <p>The raw EEM mean map is dominated by very strong near-diagonal and saturated regions. Those features are visually impressive but scientifically risky to use directly because they are strongly confounded by detector saturation and optical scatter. The cleaned EEM panel below is therefore the more appropriate map for feature selection.</p>
    </div>

    <div class="card figure">
      <h2>Raman Mean Spectrum</h2>
      <p>Annotated peaks are dominant local maxima in the average Raman signature. They provide a first-pass set of feature windows for peak intensity, peak ratio, or PLS loading interpretation.</p>
      {raman_svg}
      <ul>{peak_items}</ul>
    </div>

    <div class="card figure">
      <h2>Raw EEM Mean Map</h2>
      <p>This panel shows the dataset-average fluorescence landscape before masking. The largest hotspots include heavily saturated or near-diagonal regions and should not automatically be interpreted as reliable biochemical features.</p>
      {eem_svg}
      <ul>{raw_hotspots}</ul>
    </div>

    <div class="card figure">
      <h2>Cleaned EEM Mean Map</h2>
      <p>This second-pass panel masks cells with more than 5% <code>OVER</code> values and suppresses near-diagonal scatter by removing cells where emission is within 20 nm of excitation. These annotated hotspots are the more defensible fluorescence candidates for machine learning.</p>
      {eem_clean_svg}
      <ul>{clean_hotspots}</ul>
    </div>

    <div class="card figure">
      <h2>EEM Saturation Diagnostics</h2>
      <p>The saturation map shows where detector nonlinearity is concentrated. Regions with high <code>OVER</code> frequency should be masked, imputed carefully, or excluded during model fitting.</p>
      {over_svg}
    </div>

    <div class="card">
      <h2>Recommended ML Use</h2>
      <div class="two-col">
        <div>
          <p><strong>Raman</strong></p>
          <ul>
            <li>Use the full cropped spectrum from 500-2000 cm<sup>-1</sup> for baseline PLS.</li>
            <li>Track the annotated peak regions as interpretable checkpoints.</li>
            <li>Compare full-spectrum PLS with peak-window-only models.</li>
          </ul>
        </div>
        <div>
          <p><strong>EEM</strong></p>
          <ul>
            <li>Do not train directly on the unmasked raw mean hotspot regions.</li>
            <li>Mask saturated and near-diagonal scatter regions first.</li>
            <li>Use the cleaned hotspot map to guide wavelength-pair selection and feature importance review.</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def write_report(eem_data, raman_data, raman_peaks, eem_peaks, out_path):
    lines = [
        "# Raman and EEM Visual Summary",
        "",
        f"- Raman files analysed: {raman_data['files']}",
        f"- EEM files analysed: {eem_data['files']}",
        "- Source folders: `Raw_data_Plate` and `Raw_data_Flask` only; `Raw_data_files_ALL` was excluded to avoid duplication.",
        "",
        "## Raman dominant peaks in the mean spectrum",
        "",
    ]
    for peak_x, peak_y, _ in raman_peaks:
        lines.append(f"- `{peak_x:.0f} cm^-1`: dominant mean-spectrum maximum, intensity `{peak_y:.0f}`")
    lines += [
        "",
        "## EEM dominant hotspots in the mean matrix",
        "",
    ]
    for value, ex, em, _, _ in eem_peaks:
        lines.append(f"- `Ex {int(ex)} nm / Em {int(em)} nm`: mean intensity `{value:.0f}`")
    max_over = max(v for row in eem_data["over_fraction"] for v in row)
    lines += [
        "",
        "## Interpretation notes",
        "",
        "- Raman annotations mark the strongest local maxima in the dataset-average spectrum. They are useful candidate regions for feature engineering and peak-ratio analysis.",
        "- EEM hotspots identify excitation-emission coordinates with consistently strong fluorescence signal across the dataset.",
        f"- The maximum EEM detector saturation fraction observed locally is `{max_over:.2f}`; saturated regions should be masked or handled explicitly in downstream modeling.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eem_files = load_eem_files()
    raman_files = load_raman_files()
    eem_data = aggregate_eem(eem_files)
    raman_data = aggregate_raman(raman_files)

    raman_peaks = local_maxima_1d(raman_data["shift"], raman_data["mean"], min_spacing=90.0, top_n=6)
    eem_peaks = hotspot_maxima_2d(eem_data["excitation"], eem_data["emission"], eem_data["mean_matrix"], top_n=6)
    eem_clean = cleaned_eem_matrix(
        eem_data["excitation"],
        eem_data["emission"],
        eem_data["mean_matrix"],
        eem_data["over_fraction"],
        over_threshold=0.05,
        scatter_margin=20.0,
    )
    eem_clean_peaks = hotspot_maxima_2d_masked(
        eem_data["excitation"],
        eem_data["emission"],
        eem_clean,
        top_n=6,
    )

    render_raman_mean_svg(raman_data, raman_peaks, OUT_DIR / "raman_mean_annotated.svg")
    render_raman_heatmap_svg(raman_data, OUT_DIR / "raman_all_heatmap.svg")
    render_eem_heatmap_svg(eem_data, eem_peaks, OUT_DIR / "eem_mean_annotated.svg")
    render_eem_over_svg(eem_data, OUT_DIR / "eem_over_fraction.svg")
    render_eem_cleaned_svg(eem_data, eem_clean, eem_clean_peaks, OUT_DIR / "eem_cleaned_annotated.svg")
    write_report(eem_data, raman_data, raman_peaks, eem_peaks, OUT_DIR / "visual_summary.md")
    write_html_report(
        eem_data,
        raman_data,
        raman_peaks,
        eem_peaks,
        eem_clean_peaks,
        OUT_DIR / "visual_report.html",
    )

    print(f"Output directory: {OUT_DIR}")
    print(f"Raman files analysed: {raman_data['files']}")
    print(f"EEM files analysed: {eem_data['files']}")


if __name__ == "__main__":
    main()
