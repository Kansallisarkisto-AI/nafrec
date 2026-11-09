"""
PAGE XML (2013-07-15) writer for the HTR and OCR components.
Page format: see the module docstring of alto_writer.py.

    write_page_xml(page, path)
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


def build_page_xml(page):
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

    # Page
    page_el = etree.SubElement(root, "Page")
    page_el.set("imageFilename", str(page["image_path"]))
    if page.get("width") is not None and page.get("height") is not None:
        page_el.set("imageWidth", str(int(page["width"])))
        page_el.set("imageHeight", str(int(page["height"])))
    else:
        logging.warning(f"Image size unknown for {page['image_path']}: imageWidth/imageHeight left out")

    for i, region in enumerate(page.get("regions") or []):
        region_id = f"r{i}"
        tr = etree.SubElement(page_el, "TextRegion")
        tr.set("id", region_id)
        tr.set("custom", f"readingOrder {{index:{i};}}")
        etree.SubElement(tr, "Coords").set("points", format_polygon(region["polygon"]))

        for j, line in enumerate(region["lines"]):
            tl = etree.SubElement(tr, "TextLine")
            tl.set("id", f"{region_id}l{j}")
            tl.set("custom", f"readingOrder {{index:{j};}}")
            etree.SubElement(tl, "Coords").set("points", format_polygon(line["polygon"]))
            te = etree.SubElement(tl, "TextEquiv")
            etree.SubElement(te, "Unicode").text = (line.get("text") or "").strip()

    return etree.ElementTree(root)


def write_page_xml(page, path):
    build_page_xml(page).write(path, xml_declaration=True, encoding="utf-8", method="xml", pretty_print=True)
    logging.info(f"PAGE XML saved to {path}")