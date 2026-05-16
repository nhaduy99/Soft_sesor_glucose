import csv
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"a": NS_MAIN, "r": NS_REL}


BASE_DIR = Path(
    r"C:\Users\nhadu\OneDrive - UTS\C3_UTS\Soft_Sensor_Loc\Emilie data\Raw_data\Emilie_SoftSensor"
)
CODEX_DIR = BASE_DIR / "Codex_inventory"
XLSX_DIR = BASE_DIR / "__Emilie_SoftSensor"

INVENTORY_CSV = CODEX_DIR / "eem_raman_hplc_inventory.csv"
OUT_ENRICHED = CODEX_DIR / "eem_raman_hplc_inventory_enriched.csv"
OUT_SUMMARY = CODEX_DIR / "eem_raman_hplc_inventory_enriched_summary.csv"

METADATA_XLSX = XLSX_DIR / "metadata_Emilie_SoftSensor_Rhamnose_Plate.xlsx"
LEGEND_XLSX = XLSX_DIR / "Experiment legend.xlsx"
HPLC_LEGEND_XLSX = XLSX_DIR / "Emilie_HPLC_Sample legend.xlsx"


def read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    strings = []
    for si in root.findall(f"{{{NS_MAIN}}}si"):
        strings.append("".join(t.text or "" for t in si.iter(f"{{{NS_MAIN}}}t")))
    return strings


def get_sheet_target(zf: zipfile.ZipFile, sheet_name: str) -> str:
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    for sheet in wb.find("a:sheets", NS):
        if sheet.attrib.get("name") == sheet_name:
            rel_id = sheet.attrib[f"{{{NS_REL}}}id"]
            return "xl/" + relmap[rel_id].lstrip("/")
    raise KeyError(f"Sheet not found: {sheet_name}")


def read_sheet_rows(path: Path, sheet_name: str) -> list[list[str]]:
    with zipfile.ZipFile(path) as zf:
        shared_strings = read_shared_strings(zf)
        target = get_sheet_target(zf, sheet_name)
        root = ET.fromstring(zf.read(target))
        rows = []
        for row in root.findall(".//a:sheetData/a:row", NS):
            values = []
            for cell in row.findall("a:c", NS):
                cell_type = cell.attrib.get("t")
                value_node = cell.find("a:v", NS)
                value = value_node.text if value_node is not None else ""
                if cell_type == "s" and value:
                    value = shared_strings[int(value)]
                values.append(value)
            rows.append(values)
        return rows


def rows_to_dicts(rows: list[list[str]]) -> list[dict[str, str]]:
    header = rows[0]
    output = []
    for raw_row in rows[1:]:
        row = list(raw_row) + [""] * max(0, len(header) - len(raw_row))
        output.append({header[i]: row[i] for i in range(len(header))})
    return output


def parse_plate_spec(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    nums = re.findall(r"\d+", text)
    if "-" in text and len(nums) == 2:
        start, end = int(nums[0]), int(nums[1])
        return [str(i) for i in range(start, end + 1)]
    return nums


def parse_experiment_legend(path: Path) -> dict[tuple[str, str, str], str]:
    mapping: dict[tuple[str, str, str], str] = {}
    for exp_num in ("1", "2", "3", "4"):
        rows = read_sheet_rows(path, f"Exp{exp_num.zfill(3)}")
        current_plates: list[str] = []
        header_cols: list[str] = []
        for row in rows:
            if not row:
                continue
            title = row[0].strip()
            if title.startswith("Plate"):
                current_plates = parse_plate_spec(title)
                header_cols = row[1:]
                continue
            row_label = title
            if row_label not in list("ABCDEFGH"):
                continue
            for idx, col_label in enumerate(header_cols, start=1):
                if not col_label:
                    continue
                value = row[idx].strip() if idx < len(row) else ""
                if not value:
                    continue
                well = f"{row_label}{col_label}"
                for plate in current_plates:
                    mapping[(exp_num, plate, well)] = value
    return mapping


def parse_hplc_legend(path: Path) -> dict[str, dict[str, str]]:
    rows = read_sheet_rows(path, "Samples for HPLC")
    header = rows[0]
    normalized = []
    for i, col in enumerate(header):
        col = col.strip()
        if col:
            normalized.append(col)
        else:
            normalized.append(f"Extra_{i+1}")
    records = {}
    for raw_row in rows[1:]:
        row = list(raw_row) + [""] * max(0, len(normalized) - len(raw_row))
        record = {normalized[i]: row[i] for i in range(len(normalized))}
        sample = record.get("Samples", "").strip()
        if sample:
            records[sample] = record
    return records


def load_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def basename(path_text: str) -> str:
    return Path(path_text).name.lower() if path_text else ""


def extract_hplc_code(*candidates: str) -> str:
    for text in candidates:
        if not text:
            continue
        match = re.search(r"\b(N\d|S\d|B\d|STD)\b", text)
        if match:
            return match.group(1)
    return ""


def build_metadata_map(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    eem_rows = rows_to_dicts(read_sheet_rows(path, "EEM_Metadata_Plate"))
    raman_rows = rows_to_dicts(read_sheet_rows(path, "Raman_Metadata_Plate"))
    return (
        {row["FileName"].lower(): row for row in eem_rows if row.get("FileName")},
        {row["FileName"].lower(): row for row in raman_rows if row.get("FileName")},
    )


def bool_str(value: bool) -> str:
    return "True" if value else "False"


def build_enriched_inventory() -> list[dict[str, str]]:
    inventory_rows = load_inventory(INVENTORY_CSV)
    eem_meta_by_file, raman_meta_by_file = build_metadata_map(METADATA_XLSX)
    exp_legend_map = parse_experiment_legend(LEGEND_XLSX)
    hplc_legend_map = parse_hplc_legend(HPLC_LEGEND_XLSX)

    enriched = []
    for row in inventory_rows:
        eem_name = basename(row.get("eem_file", ""))
        raman_name = basename(row.get("raman_file", ""))

        eem_meta = eem_meta_by_file.get(eem_name)
        raman_meta = raman_meta_by_file.get(raman_name)
        primary_meta = eem_meta or raman_meta or {}

        experiment = primary_meta.get("Experiment", "")
        plate = primary_meta.get("Plate", "")
        well = primary_meta.get("Well", row.get("sample_id", ""))
        legend_label = exp_legend_map.get((experiment, plate, well), "")
        metadata_media = primary_meta.get("Media", "")
        hplc_sample_code = extract_hplc_code(legend_label, metadata_media)
        hplc_record = hplc_legend_map.get(hplc_sample_code, {})

        enriched_row = dict(row)
        enriched_row.update(
            {
                "plate_metadata_present": bool_str(bool(primary_meta)),
                "eem_plate_metadata_present": bool_str(bool(eem_meta)),
                "raman_plate_metadata_present": bool_str(bool(raman_meta)),
                "metadata_experiment": experiment,
                "metadata_plate": plate,
                "metadata_well": well,
                "metadata_media": metadata_media,
                "legend_treatment_label": legend_label,
                "metadata_strain": primary_meta.get("Strain", ""),
                "metadata_date": primary_meta.get("Date", ""),
                "metadata_comment": primary_meta.get("Comment", ""),
                "metadata_culture_duration_hours": primary_meta.get("CultureDuration_hours", ""),
                "metadata_od750": primary_meta.get("OD750", ""),
                "metadata_support": primary_meta.get("Support", ""),
                "metadata_data_type": primary_meta.get("DataType", ""),
                "hplc_sample_code": hplc_sample_code,
                "hplc_sample_legend_present": bool_str(bool(hplc_record)),
                "hplc_sample_date": hplc_record.get("Date", ""),
                "hplc_total_volume_centrifuged_ml_per_falcon": hplc_record.get(
                    "Total volume centrifuged before drying (mL per falcon)", ""
                ),
                "hplc_number_of_falcons": hplc_record.get("No. of Falcons", ""),
                "hplc_legend_note_1": hplc_record.get("Extra_5", ""),
                "hplc_legend_note_2": hplc_record.get("Extra_6", ""),
                "hplc_quantitative_reference_present": "False",
                "hplc_quantitative_reference_status": "No quantitative HPLC reference table found in current folder",
            }
        )
        enriched.append(enriched_row)

    return enriched


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    def count_where(predicate):
        return sum(1 for row in rows if predicate(row))

    summary = [
        {
            "metric": "total_inventory_rows",
            "value": str(len(rows)),
            "note": "",
        },
        {
            "metric": "rows_with_any_plate_metadata",
            "value": str(count_where(lambda r: r["plate_metadata_present"] == "True")),
            "note": "Plate metadata workbook only; flask rows remain unmatched",
        },
        {
            "metric": "rows_with_eem_plate_metadata",
            "value": str(count_where(lambda r: r["eem_plate_metadata_present"] == "True")),
            "note": "",
        },
        {
            "metric": "rows_with_raman_plate_metadata",
            "value": str(count_where(lambda r: r["raman_plate_metadata_present"] == "True")),
            "note": "",
        },
        {
            "metric": "rows_with_legend_treatment_label",
            "value": str(count_where(lambda r: bool(r["legend_treatment_label"]))),
            "note": "Derived from Experiment legend.xlsx by Experiment + Plate + Well",
        },
        {
            "metric": "rows_with_hplc_sample_code",
            "value": str(count_where(lambda r: bool(r["hplc_sample_code"]))),
            "note": "Codes like N0, S1, B2, STD",
        },
        {
            "metric": "rows_with_hplc_sample_legend_match",
            "value": str(count_where(lambda r: r["hplc_sample_legend_present"] == "True")),
            "note": "Matched to Emilie_HPLC_Sample legend.xlsx",
        },
        {
            "metric": "rows_with_quantitative_hplc_reference",
            "value": str(count_where(lambda r: r["hplc_quantitative_reference_present"] == "True")),
            "note": "Expected to remain zero until actual HPLC results are provided",
        },
    ]
    return summary


def main() -> None:
    enriched_rows = build_enriched_inventory()
    write_csv(OUT_ENRICHED, enriched_rows)
    write_csv(OUT_SUMMARY, build_summary(enriched_rows))
    print(f"Wrote {OUT_ENRICHED}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Rows: {len(enriched_rows)}")


if __name__ == "__main__":
    main()
