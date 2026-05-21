import argparse
import csv
import hashlib
import html
import json
import os
import re
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_PDF_DIR = ROOT / "literature" / "papers"
DEFAULT_SEED_CSV = ROOT / "literature" / "seed_references.csv"
DEFAULT_OUT_DIR = ROOT / "literature_review"

EVIDENCE_CSV = "rhamnose_method_evidence.csv"
REPORT_HTML = "rhamnose_method_review.html"
SCREENED_CSV = "screened_references.csv"
LOG_CSV = "extraction_log.csv"

EVIDENCE_FIELDS = [
    "paper_id",
    "title",
    "year",
    "doi",
    "source_type",
    "analyte",
    "matrix_or_sample",
    "detection_method",
    "modelling_method",
    "input_features",
    "preprocessing",
    "validation_design",
    "metrics",
    "key_result",
    "limitations",
    "relevance_to_project",
    "supporting_quote",
    "page_or_section",
    "confidence",
]

SCREENED_FIELDS = [
    "paper_id",
    "title",
    "year",
    "doi",
    "url",
    "source_type",
    "source_path",
    "status",
    "reason",
]

LOG_FIELDS = ["timestamp", "level", "stage", "source", "message"]

DETECTION_PATTERNS = {
    "Raman": [r"\braman\b", r"\bsers\b", r"surface[- ]enhanced raman"],
    "EEM": [r"\beem\b", r"excitation[- ]emission", r"fluorescence"],
    "HPLC": [r"\bhplc\b", r"high[- ]performance liquid chromatography"],
    "LC-MS": [r"\blc[- ]?ms\b", r"liquid chromatography[- ]mass"],
    "NMR": [r"\bnmr\b", r"nuclear magnetic resonance"],
    "enzymatic assay": [r"enzymatic assay", r"enzyme assay", r"enzyme[- ]based"],
}

MODEL_PATTERNS = {
    "PLSR": [r"\bplsr\b", r"partial least squares", r"\bpls\b"],
    "PCR": [r"\bpcr\b", r"principal component regression"],
    "ridge": [r"ridge regression", r"\bridge\b"],
    "SVR": [r"\bsvr\b", r"support vector regression", r"support vector machine"],
    "random forest": [r"random forest", r"\brf\b"],
    "XGBoost": [r"xgboost", r"gradient boosting", r"boosted tree"],
    "ANN/MLP": [r"\bann\b", r"neural network", r"\bmlp\b", r"deep learning"],
    "PARAFAC": [r"parafac", r"parallel factor"],
    "PCA": [r"\bpca\b", r"principal component analysis"],
    "kNN": [r"\bknn\b", r"k-nearest", r"nearest neighbo"],
}

PREPROCESSING_PATTERNS = {
    "baseline correction": [r"baseline correction", r"baseline corrected", r"asymmetric least squares", r"\bals\b"],
    "smoothing": [r"smoothing", r"savitzky", r"sgolay", r"moving average"],
    "normalization": [r"normalization", r"normalisation", r"normalized", r"normalised"],
    "derivatives": [r"derivative", r"first derivative", r"second derivative"],
    "scatter masking": [r"scatter mask", r"rayleigh", r"raman scatter"],
    "SNV": [r"\bsnv\b", r"standard normal variate"],
    "mean centering": [r"mean centering", r"mean-centering", r"autoscal"],
}

FEATURE_PATTERNS = {
    "spectra": [r"spectra", r"spectrum", r"spectral"],
    "bands": [r"band", r"wavenumber", r"cm-1", r"cm\^-1"],
    "EEM matrix": [r"eem matrix", r"excitation[- ]emission matrix", r"fluorescence matrix"],
    "PARAFAC scores": [r"parafac score", r"component score", r"scores"],
    "chromatographic peaks": [r"chromatographic peak", r"peak area", r"retention time"],
    "fused features": [r"fusion", r"fused", r"multimodal", r"multi-modal"],
}

VALIDATION_PATTERNS = {
    "train/test": [r"train/test", r"training set", r"test set", r"hold[- ]out"],
    "cross-validation": [r"cross[- ]validation", r"\bcv\b", r"leave[- ]one[- ]out", r"loo"],
    "grouped split": [r"grouped", r"batch split", r"experiment split", r"independent batch"],
    "external validation": [r"external validation", r"independent validation", r"validation set"],
    "calibration only": [r"calibration model", r"calibration set", r"calibration curve"],
}

MATRIX_PATTERNS = {
    "standard solution": [r"standard solution", r"standard sample", r"calibration standard"],
    "fermentation broth": [r"fermentation", r"broth", r"bioreactor"],
    "algae/culture": [r"algae", r"microalgae", r"culture sample", r"cell culture"],
    "food": [r"food", r"juice", r"wine", r"honey", r"milk"],
    "biological": [r"serum", r"urine", r"plasma", r"tissue", r"biological"],
}

METRIC_REGEX = re.compile(
    r"\b(RMSE|R2|R\^2|MAE|MSE|accuracy|LOD|LOQ|recovery|correlation|r\s*=|RPD|SEP|SEC)\b"
    r"[^.;:\n]{0,100}",
    re.IGNORECASE,
)
DOI_REGEX = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
YEAR_REGEX = re.compile(r"\b(19|20)\d{2}\b")


def now_stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def normalize_space(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def paper_id_for(*parts):
    raw = "|".join(normalize_space(part).lower() for part in parts if part)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"lit_{digest}"


def log_entry(logs, level, stage, source, message):
    logs.append(
        {
            "timestamp": now_stamp(),
            "level": level,
            "stage": stage,
            "source": str(source),
            "message": message,
        }
    )


def read_csv_rows(path, logs):
    if not path.exists():
        log_entry(logs, "INFO", "seed", path, "Seed CSV not found; continuing without seed references.")
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    log_entry(logs, "INFO", "seed", path, f"Loaded {len(rows)} seed references.")
    return rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def find_matches(text, pattern_map):
    found = []
    for label, patterns in pattern_map.items():
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
            found.append(label)
    return found


def first_match(text, pattern_map, default="unknown"):
    matches = find_matches(text, pattern_map)
    return matches[0] if matches else default


def extract_doi(text, fallback=""):
    return normalize_space(fallback) or (DOI_REGEX.search(text).group(0) if DOI_REGEX.search(text) else "")


def extract_year(text, fallback=""):
    if fallback:
        return str(fallback)
    match = YEAR_REGEX.search(text)
    return match.group(0) if match else ""


def sentence_split(text):
    text = normalize_space(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [part.strip() for part in parts if len(part.strip()) > 20]


def choose_quote(sentences):
    preferred = []
    for sent in sentences:
        low = sent.lower()
        score = 0
        if "rhamnose" in low or "rha" in low:
            score += 5
        if any(term in low for term in ["raman", "hplc", "fluorescence", "eem", "parafac"]):
            score += 3
        if any(term in low for term in ["model", "regression", "validation", "calibration", "prediction"]):
            score += 2
        if METRIC_REGEX.search(sent):
            score += 2
        if score:
            preferred.append((score, len(sent), sent))
    if preferred:
        preferred.sort(key=lambda item: (-item[0], item[1]))
        return preferred[0][2][:500]
    return sentences[0][:500] if sentences else ""


def choose_key_result(sentences):
    metric_sentences = [sent for sent in sentences if METRIC_REGEX.search(sent)]
    if metric_sentences:
        return metric_sentences[0][:500]
    result_terms = ["result", "predict", "determin", "detect", "quantif", "calibration"]
    for sent in sentences:
        if any(term in sent.lower() for term in result_terms):
            return sent[:500]
    return ""


def choose_limitations(sentences):
    limit_terms = ["limit", "weak", "interference", "matrix", "future", "only", "however", "challenge"]
    for sent in sentences:
        if any(term in sent.lower() for term in limit_terms):
            return sent[:500]
    return ""


def metric_summary(text):
    metrics = []
    for match in METRIC_REGEX.finditer(text):
        metrics.append(normalize_space(match.group(0)))
    return "; ".join(dict.fromkeys(metrics))[:800]


def infer_relevance(detections, models, matrix, text):
    low = text.lower()
    score = 0
    if "rhamnose" in low:
        score += 3
    if "Raman" in detections:
        score += 2
    if "EEM" in detections or "PARAFAC" in models:
        score += 2
    if "HPLC" in detections:
        score += 2
    if any(model in models for model in ["PLSR", "SVR", "random forest", "XGBoost", "ANN/MLP"]):
        score += 1
    if matrix in {"fermentation broth", "algae/culture", "standard solution"}:
        score += 1
    if score >= 6:
        return "high"
    if score >= 3:
        return "medium"
    return "low"


def confidence_score(text, detections, models, metrics, source_type):
    score = 0.25
    if "rhamnose" in text.lower():
        score += 0.2
    if detections and detections != ["other"]:
        score += 0.15
    if models and models != ["other"]:
        score += 0.15
    if metrics:
        score += 0.1
    if source_type == "pdf":
        score += 0.1
    return f"{min(score, 0.95):.2f}"


def is_relevant_text(text):
    low = text.lower()
    if "rhamnose" not in low and "rha" not in low:
        return False
    method_hit = any(find_matches(text, mapping) for mapping in [DETECTION_PATTERNS, MODEL_PATTERNS])
    return method_hit or any(term in low for term in ["detect", "model", "predict", "quantif", "calibration"])


def extract_title_from_text(text, fallback=""):
    if fallback:
        return normalize_space(fallback)
    for line in str(text or "").splitlines():
        clean = normalize_space(line)
        if 20 <= len(clean) <= 180 and not DOI_REGEX.search(clean):
            return clean
    return "Untitled reference"


def pdf_pages(path, logs):
    try:
        import fitz  # type: ignore
    except Exception as exc:
        log_entry(logs, "WARNING", "pdf", path, f"PyMuPDF unavailable; cannot extract PDF text: {exc}")
        return [], {}
    try:
        doc = fitz.open(path)
    except Exception as exc:
        log_entry(logs, "ERROR", "pdf", path, f"Could not open PDF: {exc}")
        return [], {}
    metadata = dict(doc.metadata or {})
    pages = []
    for index, page in enumerate(doc, start=1):
        try:
            text = page.get_text("text")
        except Exception as exc:
            log_entry(logs, "WARNING", "pdf", path, f"Could not extract page {index}: {exc}")
            text = ""
        if normalize_space(text):
            pages.append({"page_or_section": f"page {index}", "text": text})
    doc.close()
    log_entry(logs, "INFO", "pdf", path, f"Extracted text from {len(pages)} pages.")
    return pages, metadata


def chunk_text_units(units, max_chars=2800):
    chunks = []
    for unit in units:
        text = normalize_space(unit.get("text", ""))
        section = unit.get("page_or_section", "")
        if len(text) <= max_chars:
            chunks.append({"page_or_section": section, "text": text})
            continue
        sentences = sentence_split(text)
        current = []
        current_len = 0
        for sent in sentences:
            if current_len + len(sent) > max_chars and current:
                chunks.append({"page_or_section": section, "text": " ".join(current)})
                current = []
                current_len = 0
            current.append(sent)
            current_len += len(sent) + 1
        if current:
            chunks.append({"page_or_section": section, "text": " ".join(current)})
    return chunks


def build_reference_from_pdf(path, logs):
    pages, metadata = pdf_pages(path, logs)
    combined = "\n".join(page["text"] for page in pages[:3])
    title = extract_title_from_text(combined, metadata.get("title", ""))
    doi = extract_doi(combined)
    year = extract_year(combined, metadata.get("creationDate", ""))
    paper_id = paper_id_for(path.name, doi, title)
    return {
        "paper_id": paper_id,
        "title": title,
        "year": year,
        "doi": doi,
        "url": "",
        "source_type": "pdf",
        "source_path": str(path),
        "units": pages,
    }


def build_reference_from_seed(row):
    title = normalize_space(row.get("title") or row.get("Title") or row.get("name") or "")
    doi = normalize_space(row.get("doi") or row.get("DOI") or "")
    url = normalize_space(row.get("url") or row.get("URL") or "")
    year = normalize_space(row.get("year") or row.get("Year") or "")
    notes = normalize_space(row.get("notes") or row.get("Notes") or row.get("abstract") or row.get("Abstract") or "")
    source_text = " ".join(part for part in [title, notes, doi, url] if part)
    source_type = "doi" if doi else "web"
    return {
        "paper_id": paper_id_for(title, doi, url),
        "title": title or "Untitled seed reference",
        "year": year or extract_year(source_text),
        "doi": doi or extract_doi(source_text),
        "url": url,
        "source_type": source_type,
        "source_path": "",
        "units": [{"page_or_section": "seed reference", "text": source_text}],
    }


def crossref_query(query, rows, logs, limit=5):
    encoded = urllib.parse.urlencode(
        {
            "query.bibliographic": query,
            "rows": str(limit),
            "select": "DOI,title,published-print,published-online,container-title,URL,abstract",
        }
    )
    url = f"https://api.crossref.org/works?{encoded}"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "CodexRhamnoseLiteratureTool/1.0"})
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        log_entry(logs, "WARNING", "web", query, f"Crossref lookup unavailable: {exc}")
        return []
    items = payload.get("message", {}).get("items", [])
    references = []
    for item in items:
        title = normalize_space(" ".join(item.get("title", [])[:1]))
        doi = normalize_space(item.get("DOI", ""))
        item_url = normalize_space(item.get("URL", ""))
        year = ""
        for key in ["published-print", "published-online"]:
            parts = item.get(key, {}).get("date-parts", [])
            if parts and parts[0]:
                year = str(parts[0][0])
                break
        abstract = re.sub(r"<[^>]+>", " ", item.get("abstract", ""))
        text = " ".join(part for part in [title, abstract, doi] if part)
        ref = {
            "paper_id": paper_id_for(title, doi, item_url),
            "title": title or "Untitled web reference",
            "year": year,
            "doi": doi,
            "url": item_url,
            "source_type": "web",
            "source_path": "",
            "units": [{"page_or_section": "Crossref metadata", "text": text}],
        }
        references.append(ref)
    rows.extend(references)
    log_entry(logs, "INFO", "web", query, f"Crossref returned {len(references)} references.")
    return references


def llm_refine_record(record, text, logs):
    if os.environ.get("ENABLE_LLM_EXTRACTION", "").strip().lower() not in {"1", "true", "yes"}:
        return record
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        log_entry(logs, "INFO", "llm", record.get("paper_id", ""), "LLM extraction requested but OPENAI_API_KEY is not set.")
        return record
    # The tool is intentionally conservative: it leaves deterministic fields intact
    # unless a compatible endpoint returns valid JSON with matching keys.
    endpoint = os.environ.get("OPENAI_CHAT_COMPLETIONS_URL", "https://api.openai.com/v1/chat/completions")
    model = os.environ.get("OPENAI_LITERATURE_MODEL", "gpt-4o-mini")
    prompt = {
        "role": "user",
        "content": (
            "Extract rhamnose detection/modelling methods from this text as JSON using only these keys: "
            + ", ".join(EVIDENCE_FIELDS)
            + ". Keep supporting_quote under 80 words. Text:\n"
            + text[:5000]
        ),
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return one compact JSON object. Do not invent missing details."},
            prompt,
        ],
        "temperature": 0,
    }
    try:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            reply = json.loads(response.read().decode("utf-8"))
        content = reply["choices"][0]["message"]["content"]
        proposed = json.loads(content)
    except Exception as exc:
        log_entry(logs, "WARNING", "llm", record.get("paper_id", ""), f"LLM extraction failed; using rules: {exc}")
        return record
    for key in EVIDENCE_FIELDS:
        value = normalize_space(proposed.get(key, ""))
        if value and key not in {"paper_id", "source_type"}:
            record[key] = value
    log_entry(logs, "INFO", "llm", record.get("paper_id", ""), "LLM refinement applied.")
    return record


def extract_evidence_from_reference(ref, logs):
    evidence = []
    units = chunk_text_units(ref.get("units", []))
    for unit in units:
        text = normalize_space(unit.get("text", ""))
        if not is_relevant_text(text):
            continue
        detections = find_matches(text, DETECTION_PATTERNS) or ["other"]
        models = find_matches(text, MODEL_PATTERNS) or ["other"]
        preprocess = find_matches(text, PREPROCESSING_PATTERNS)
        features = find_matches(text, FEATURE_PATTERNS)
        validation = find_matches(text, VALIDATION_PATTERNS)
        matrix = first_match(text, MATRIX_PATTERNS, "unknown")
        metrics = metric_summary(text)
        sentences = sentence_split(text)
        quote = choose_quote(sentences)
        result = choose_key_result(sentences)
        limits = choose_limitations(sentences)
        relevance = infer_relevance(detections, models, matrix, text)
        confidence = confidence_score(text, detections, models, metrics, ref["source_type"])
        for detection in detections[:3]:
            for model in models[:3]:
                record = {
                    "paper_id": ref["paper_id"],
                    "title": ref.get("title", ""),
                    "year": ref.get("year", ""),
                    "doi": ref.get("doi", ""),
                    "source_type": ref.get("source_type", ""),
                    "analyte": "rhamnose",
                    "matrix_or_sample": matrix,
                    "detection_method": detection,
                    "modelling_method": model,
                    "input_features": "; ".join(features) if features else "not specified",
                    "preprocessing": "; ".join(preprocess) if preprocess else "not specified",
                    "validation_design": "; ".join(validation) if validation else "not specified",
                    "metrics": metrics,
                    "key_result": result,
                    "limitations": limits,
                    "relevance_to_project": relevance,
                    "supporting_quote": quote,
                    "page_or_section": unit.get("page_or_section", ""),
                    "confidence": confidence,
                }
                evidence.append(llm_refine_record(record, text, logs))
    return evidence


def dedupe_references(references):
    seen = set()
    deduped = []
    for ref in references:
        key = (ref.get("doi", "").lower(), ref.get("title", "").lower(), ref.get("source_path", "").lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped


def dedupe_evidence(rows):
    seen = set()
    deduped = []
    for row in rows:
        key = (
            row.get("paper_id", ""),
            row.get("detection_method", ""),
            row.get("modelling_method", ""),
            row.get("supporting_quote", "")[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def screen_reference(ref, evidence_count):
    title_text = " ".join(
        [ref.get("title", ""), ref.get("doi", ""), " ".join(unit.get("text", "") for unit in ref.get("units", [])[:2])]
    )
    if evidence_count:
        return "included", f"{evidence_count} method evidence rows extracted"
    if is_relevant_text(title_text):
        return "screened_no_method_row", "Rhamnose-relevant text found but no method row passed extraction"
    return "excluded", "No rhamnose method/detection evidence found"


def html_table(rows, fields, limit=None):
    shown = rows[:limit] if limit else rows
    if not shown:
        return "<p class=\"empty\">No rows found.</p>"
    head = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body_rows = []
    for row in shown:
        cells = "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def grouped_counts(rows, field):
    counter = Counter(row.get(field, "unknown") or "unknown" for row in rows)
    return [{"category": key, "count": value} for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]


def generate_report(out_dir, evidence_rows, screened_rows, logs):
    high_rows = [row for row in evidence_rows if row.get("relevance_to_project") == "high"]
    medium_rows = [row for row in evidence_rows if row.get("relevance_to_project") == "medium"]
    detection_counts = grouped_counts(evidence_rows, "detection_method")
    model_counts = grouped_counts(evidence_rows, "modelling_method")
    relevance_counts = grouped_counts(evidence_rows, "relevance_to_project")
    css = """
    body { font-family: Arial, sans-serif; color: #1f2933; margin: 32px; line-height: 1.45; }
    h1, h2, h3 { color: #102a43; }
    .note { background: #f0f4f8; border-left: 4px solid #486581; padding: 12px 16px; }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
    .card { border: 1px solid #d9e2ec; border-radius: 6px; padding: 14px; background: #ffffff; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }
    th, td { border: 1px solid #d9e2ec; padding: 7px 8px; vertical-align: top; }
    th { background: #f0f4f8; text-align: left; }
    .empty { color: #627d98; font-style: italic; }
    code { background: #f0f4f8; padding: 1px 4px; border-radius: 3px; }
    """
    summary_cards = f"""
    <div class="grid">
      <div class="card"><strong>Evidence rows</strong><br>{len(evidence_rows)}</div>
      <div class="card"><strong>Screened references</strong><br>{len(screened_rows)}</div>
      <div class="card"><strong>High relevance rows</strong><br>{len(high_rows)}</div>
    </div>
    """
    project_comparison = """
    <div class="note">
      <p><strong>Project comparison.</strong> The current soft-sensor workflow treats Raman as the most direct carbohydrate signal for rhamnose, while EEM and PARAFAC are interpreted as indirect fluorescence/process-state inputs. HPLC remains the required quantitative reference method for culture samples. Literature evidence should therefore be used to justify spectroscopy preprocessing, model families, validation design, and the need for grouped culture-sample validation once quantitative HPLC targets are available.</p>
    </div>
    """
    gap_items = [
        "Prioritize citations that report rhamnose or carbohydrate quantification with Raman, HPLC, EEM/PARAFAC, or chemometric validation.",
        "Flag papers that use calibration-only validation separately from papers with external or grouped validation.",
        "Use high-relevance rows to strengthen Raman band/preprocessing discussion and model-method justification.",
        "Do not treat literature evidence as culture-sample validation for this project; measured HPLC culture targets are still required.",
    ]
    gaps = "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in gap_items) + "</ul>"
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Rhamnose Literature Method Review</title>
  <style>{css}</style>
</head>
<body>
  <h1>Rhamnose Literature Method Review</h1>
  <p>Generated {html.escape(now_stamp())}. This report extracts auditable method evidence for rhamnose modelling and detection.</p>
  {summary_cards}
  {project_comparison}

  <h2>Evidence by Detection Method</h2>
  {html_table(detection_counts, ["category", "count"])}

  <h2>Evidence by Modelling Method</h2>
  {html_table(model_counts, ["category", "count"])}

  <h2>Evidence by Project Relevance</h2>
  {html_table(relevance_counts, ["category", "count"])}

  <h2>High-Relevance Evidence</h2>
  {html_table(high_rows, EVIDENCE_FIELDS, limit=50)}

  <h2>Medium-Relevance Evidence</h2>
  {html_table(medium_rows, EVIDENCE_FIELDS, limit=50)}

  <h2>Gaps and Recommended Citation Use</h2>
  {gaps}

  <h2>Screened References</h2>
  {html_table(screened_rows, SCREENED_FIELDS, limit=100)}

  <h2>Extraction Log</h2>
  {html_table(logs[-100:], LOG_FIELDS)}
</body>
</html>
"""
    (out_dir / REPORT_HTML).write_text(report, encoding="utf-8")


def collect_references(args, logs):
    references = []
    pdf_dir = Path(args.pdf_dir)
    if pdf_dir.exists():
        pdf_paths = sorted(pdf_dir.glob("*.pdf"))
        log_entry(logs, "INFO", "pdf", pdf_dir, f"Found {len(pdf_paths)} local PDF files.")
        for path in pdf_paths:
            references.append(build_reference_from_pdf(path, logs))
    else:
        log_entry(logs, "INFO", "pdf", pdf_dir, "PDF folder not found; continuing without local PDFs.")
    for row in read_csv_rows(Path(args.seed_csv), logs):
        references.append(build_reference_from_seed(row))
    if args.web:
        queries = [
            "rhamnose Raman spectroscopy quantification chemometrics",
            "rhamnose HPLC quantification validation",
            "rhamnose fluorescence EEM PARAFAC",
            "rhamnose partial least squares regression spectroscopy",
            "rhamnose soft sensor Raman EEM",
        ]
        for query in queries:
            crossref_query(query, references, logs, limit=args.web_limit)
    else:
        log_entry(logs, "INFO", "web", "disabled", "Web discovery disabled. Use --web to query Crossref metadata.")
    return dedupe_references(references)


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    references = collect_references(args, logs)
    all_evidence = []
    evidence_by_paper = defaultdict(int)
    for ref in references:
        rows = extract_evidence_from_reference(ref, logs)
        rows = dedupe_evidence(rows)
        all_evidence.extend(rows)
        evidence_by_paper[ref["paper_id"]] += len(rows)
    all_evidence = dedupe_evidence(all_evidence)
    screened = []
    for ref in references:
        status, reason = screen_reference(ref, evidence_by_paper.get(ref["paper_id"], 0))
        screened.append(
            {
                "paper_id": ref.get("paper_id", ""),
                "title": ref.get("title", ""),
                "year": ref.get("year", ""),
                "doi": ref.get("doi", ""),
                "url": ref.get("url", ""),
                "source_type": ref.get("source_type", ""),
                "source_path": ref.get("source_path", ""),
                "status": status,
                "reason": reason,
            }
        )
    write_csv(out_dir / EVIDENCE_CSV, all_evidence, EVIDENCE_FIELDS)
    write_csv(out_dir / SCREENED_CSV, screened, SCREENED_FIELDS)
    write_csv(out_dir / LOG_CSV, logs, LOG_FIELDS)
    generate_report(out_dir, all_evidence, screened, logs)
    print(f"references_screened={len(screened)}")
    print(f"evidence_rows={len(all_evidence)}")
    print(f"wrote={out_dir / EVIDENCE_CSV}")
    print(f"wrote={out_dir / REPORT_HTML}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract auditable rhamnose literature method evidence from local PDFs, seed references, and optional web metadata."
    )
    parser.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR), help="Folder containing local PDF papers.")
    parser.add_argument("--seed-csv", default=str(DEFAULT_SEED_CSV), help="Optional seed CSV with title, doi, url, notes.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output folder for CSV and HTML reports.")
    parser.add_argument("--web", action="store_true", help="Enable optional Crossref web discovery.")
    parser.add_argument("--web-limit", type=int, default=5, help="Maximum Crossref records per query when --web is enabled.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
