"""
PAGE XML (2013-07-15) writer for the HTR and OCR components.
Page format: see the module docstring of alto_writer.py.

    write_page_xml(page, path, comments=None)
"""

import datetime
import logging

from lxml import etree

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
PAGE_XSD = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"


def format_polygon(polygon):
    """'x,y x,y ...' with non-negative integers, as the PAGE schema requires."""
    return " ".join(f"{max(int(x), 0)},{max(int(y), 0)}" for x, y in polygon)


def _region_polygon(region):
    """The region's polygon, or the bounding box of its lines; None if it has neither."""
    polygon = region.get("polygon")
    if polygon is not None and len(polygon) > 0:
        return polygon
    points = [p for line in region.get("lines") or [] for p in line["polygon"]]
    if not points:
        return None
    xs = [int(p[0]) for p in points]
    ys = [int(p[1]) for p in points]
    return [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]


def build_page_xml(page, comments=None):
    root = etree.Element(
        "PcGts",
        nsmap={None: PAGE_NS, "xsi": XSI_NS},
        attrib={etree.QName(XSI_NS, "schemaLocation"): f"{PAGE_NS} {PAGE_XSD}"},
    )

    # Metadata
    metadata = etree.SubElement(root, "Metadata")
    etree.SubElement(metadata, "Creator").text = "Kansallisarkisto - National Archives of Finland"
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    etree.SubElement(metadata, "Created").text = now
    etree.SubElement(metadata, "LastChange").text = now
    if comments:
        # PAGE schema order in Metadata: Creator, Created, LastChange, Comments
        etree.SubElement(metadata, "Comments").text = comments

    # Page
    page_el = etree.SubElement(root, "Page")
    page_el.set("imageFilename", str(page["image_path"]))
    if page.get("width") is not None and page.get("height") is not None:
        page_el.set("imageWidth", str(int(page["width"])))
        page_el.set("imageHeight", str(int(page["height"])))
    else:
        logging.warning(f"Image size unknown for {page['image_path']}: imageWidth/imageHeight left out")

    i = 0
    for region in page.get("regions") or []:
        polygon = _region_polygon(region)
        if polygon is None:  # no polygon and no lines: nothing to write
            continue
        region_id = f"r{i}"
        tr = etree.SubElement(page_el, "TextRegion")
        tr.set("id", region_id)
        tr.set("custom", f"readingOrder {{index:{i};}}")
        etree.SubElement(tr, "Coords").set("points", format_polygon(polygon))
        i += 1

        for j, line in enumerate(region["lines"]):
            tl = etree.SubElement(tr, "TextLine")
            tl.set("id", f"{region_id}l{j}")
            tl.set("custom", f"readingOrder {{index:{j};}}")
            etree.SubElement(tl, "Coords").set("points", format_polygon(line["polygon"]))
            te = etree.SubElement(tl, "TextEquiv")
            etree.SubElement(te, "Unicode").text = (line.get("text") or "").strip()

    return etree.ElementTree(root)


def write_page_xml(page, path, comments=None):
    """comments: optional free text for Metadata/Comments, e.g. a description of the models used."""
    build_page_xml(page, comments).write(path, xml_declaration=True, encoding="utf-8", method="xml", pretty_print=True)
    logging.info(f"PAGE XML saved to {path}")