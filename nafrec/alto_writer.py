"""
ALTO 4.4 writer for the HTR and OCR components.

    write_alto(page, steps, path, include_script=False)
    alto_string(page, steps, include_script=False) -> str

    steps: list of ProcessingStep, defined by the calling component
    (model names, versions and creators are not part of this module).

Page format (input of this module and of page_xml.py; built by each component,
statistics and language added with page_stats.add_page_info):

    {
        "image_path": str,
        "width": int | None,             # left out of the XML if None
        "height": int | None,
        "language": str | None,          # ISO 639-1, e.g. "fi"
        "stats": {name: value, ...},     # page_stats.compute_stats
        "regions": [                     # in reading order; [] for an empty page
            {
                "name": str,             # e.g. "paragraph", "marginalia", "text"
                "polygon": [[x, y], ...],
                "lines": [               # in reading order
                    {"polygon": [[x, y], ...], "text": str,
                     "conf": float | None, "script": str | None},
                ],
            },
        ],
    }
    Other keys in the dicts are allowed and ignored by the writers.

Statistics: the writer has no knowledge of specific statistics.
    - stats["PC"] (standard ALTO attribute) is written on Page
    - all other stats are written as name=value pairs, in the order given,
      into processingStepSettings of the TEXT_RECOGNITION step
    - None values are left out

Optional script classification (include_script=True): one OtherTag per script
value in Tags, referenced from each TextLine with TAGREFS.
"""

import logging
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
ALTO_XSD = "http://www.loc.gov/standards/alto/v4/alto-4-4.xsd"
AGENCY = "Kansallisarkisto - National Archives of Finland"
PAGE_CONF_KEY = "PC"                 # stat written as the Page PC attribute
STATS_STEP_ID = "TEXT_RECOGNITION"   # processing step that carries the other stats
CONF_DECIMALS = 4   # decimals for WC, PC and the other confidence stats

ET.register_namespace("", ALTO_NS)
ET.register_namespace("xsi", XSI_NS)

_LANG_RE = re.compile(r"^[a-zA-Z]{1,8}(-[a-zA-Z0-9]{1,8})*$")


# --------------------------------------------------------------------------
# Processing metadata
# --------------------------------------------------------------------------

@dataclass
class ProcessingStep:
    """
    One processing step in the ALTO Description, defined by the calling component.
    The stats (except PC) are written to the step whose id is STATS_STEP_ID.
    """
    id: str
    description: str
    software_name: str | None = None
    software_version: str | None = None
    software_creator: str | None = None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def make_id(*parts):
    """Valid XML ID: no spaces or special characters, does not start with a digit."""
    clean = re.sub(r"[^A-Za-z0-9_.-]", "_", "_".join(str(p) for p in parts))
    return clean if re.match(r"[A-Za-z_]", clean) else "id_" + clean


def _q(tag):
    return f"{{{ALTO_NS}}}{tag}"


def _sub(parent, tag, text=None, **attrs):
    """SubElement in the ALTO namespace; attributes with value None are left out."""
    el = ET.SubElement(parent, _q(tag), {k: str(v) for k, v in attrs.items() if v is not None})
    if text is not None:
        el.text = str(text)
    return el


def _bbox(points):
    xs = [int(p[0]) for p in points]
    ys = [int(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _box(x0, y0, x1, y1):
    return {"HPOS": x0, "VPOS": y0, "WIDTH": max(x1 - x0, 0), "HEIGHT": max(y1 - y0, 0)}


def _conf(value):
    """Confidence in [0, 1] with 4 decimals, or None if missing or not a finite number."""
    if value is None or not math.isfinite(float(value)):
        return None
    return f"{min(max(float(value), 0.0), 1.0):.{CONF_DECIMALS}f}"


def _fmt_value(value):
    if isinstance(value, bool) or isinstance(value, int):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.{CONF_DECIMALS}f}"
    return str(value)


def _shape(parent, polygon):
    _sub(_sub(parent, "Shape"), "Polygon", POINTS=" ".join(f"{int(x)},{int(y)}" for x, y in polygon))


def region_polygon(region):
    """
    The region's polygon; if it has none, the bounding box of its lines as a rectangle;
    None if it has neither (such a region is not written).
    """
    polygon = region.get("polygon")
    if polygon is not None and len(polygon) > 0:
        return polygon
    points = [p for line in region.get("lines") or [] for p in line["polygon"]]
    if not points:
        return None
    x0, y0, x1, y1 = _bbox(points)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _script_tag_id(script):
    return make_id("SCRIPT", str(script).upper())


def _valid_language(language):
    if not language:
        return None
    if not _LANG_RE.match(str(language)) or str(language).lower() == "unknown":
        logging.warning(f"Invalid language code {language!r}, LANG left out")
        return None
    return str(language)


def _stats_settings(stats):
    parts = [f"{k}={_fmt_value(v)}" for k, v in stats.items() if k != PAGE_CONF_KEY and v is not None]
    return ";".join(parts) or None


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------

def build_alto(page, steps, include_script=False):
    """Builds an ALTO 4.4 document from a page dict. Returns an ElementTree."""
    stats = page.get("stats") or {}
    regions = page.get("regions") or []
    root = ET.Element(_q("alto"), {f"{{{XSI_NS}}}schemaLocation": f"{ALTO_NS} {ALTO_XSD}"})

    # Description
    desc = _sub(root, "Description")
    _sub(desc, "MeasurementUnit", "pixel")
    _sub(_sub(desc, "sourceImageInformation"), "fileName", page["image_path"])
    settings = _stats_settings(stats)
    if settings and not any(step.id == STATS_STEP_ID for step in steps):
        logging.warning(f"No processing step with id {STATS_STEP_ID}: page statistics are not written")
    for step in steps:
        proc = _sub(desc, "Processing", ID=step.id)
        _sub(proc, "processingAgency", AGENCY)
        _sub(proc, "processingStepDescription", step.description)
        if settings and step.id == STATS_STEP_ID:
            _sub(proc, "processingStepSettings", settings)
        if step.software_name:
            sw = _sub(proc, "processingSoftware")
            if step.software_creator:
                _sub(sw, "softwareCreator", step.software_creator)
            _sub(sw, "softwareName", step.software_name)
            if step.software_version is not None:
                _sub(sw, "softwareVersion", step.software_version)

    _sub(root, "Styles")

    # Tags for script classification
    scripts = set()
    if include_script:
        scripts = {l["script"] for r in regions for l in r["lines"] if l.get("script")}
        if scripts:
            tags = _sub(root, "Tags")
            for script in sorted(scripts):
                _sub(tags, "OtherTag", ID=_script_tag_id(script), TYPE="script", LABEL=script)

    # Layout
    width, height = page.get("width"), page.get("height")
    page_el = _sub(
        _sub(root, "Layout"), "Page",
        ID=make_id("page", Path(page["image_path"]).name),
        PHYSICAL_IMG_NR=0,
        WIDTH=width,
        HEIGHT=height,
        PC=_conf(stats.get(PAGE_CONF_KEY)),
        LANG=_valid_language(page.get("language")),
    )

    points = [p for r in regions for l in r["lines"] for p in l["polygon"]]
    if points:
        x0, y0, x1, y1 = _bbox(points)
        if width and height:
            _sub(page_el, "TopMargin", ID="TM_00001", **_box(0, 0, width, y0))
            _sub(page_el, "LeftMargin", ID="LM_00001", **_box(0, y0, x0, y1))
            _sub(page_el, "RightMargin", ID="RM_00001", **_box(x1, y0, width, y1))
            _sub(page_el, "BottomMargin", ID="BM_00001", **_box(0, y1, width, height))
        print_space = _sub(page_el, "PrintSpace", ID="PS_00001", **_box(x0, y0, x1, y1))

        counts = {}
        written = 0
        for region in regions:
            polygon = region_polygon(region)
            if polygon is None:  # no polygon and no lines: nothing to write
                continue
            written += 1
            name = region.get("name") or "region"
            n = counts.get(name, 0)
            counts[name] = n + 1
            region_id = make_id(name, n)
            region_box = _box(*_bbox(polygon))

            composed = _sub(print_space, "ComposedBlock", ID=f"block_1_{i}", **region_box)
            block = _sub(composed, "TextBlock", ID=region_id, **region_box)
            _shape(block, polygon)

            for j, line in enumerate(region["lines"]):
                line_id = f"{region_id}_line_{j}"
                line_box = _box(*_bbox(line["polygon"]))
                script = line.get("script")
                tagref = _script_tag_id(script) if script in scripts else None
                text_line = _sub(block, "TextLine", ID=line_id, TAGREFS=tagref, **line_box)
                _shape(text_line, line["polygon"])
                _sub(text_line, "String", ID=f"{line_id}_s0", CONTENT=line.get("text") or "",
                     WC=_conf(line.get("conf")), **line_box)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    return tree


def alto_string(page, steps, include_script=False):
    tree = build_alto(page, steps, include_script)
    return ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True).decode("utf-8")


def write_alto(page, steps, path, include_script=False):
    build_alto(page, steps, include_script).write(path, encoding="utf-8", xml_declaration=True)
    logging.info(f"ALTO XML saved to {path}")