"""
Page-level statistics and language detection for a page dict
(page format: see the module docstring of alto_writer.py).

The statistics are defined only in compute_stats. The XML writers take the
names and values as they come: in ALTO, "PC" becomes the Page attribute and
every other entry is written to processingStepSettings.
"""

import logging
import math

import numpy as np
import py3langid as langid

# A text row longer than this many characters counts towards N_LONG_ROWTEXT
LONG_ROW_LENGTH = 100
# Languages py3langid chooses from
LANGUAGES = ["sv", "fi", "en", "de", "ru", "lv", "lt", "la", "et", "fr", "da", "no", "is", "se"]
langid.set_languages(LANGUAGES)


def detect_language(texts):
    """Returns the detected ISO 639-1 language code, or None if there is no text with letters."""
    combined = " ".join(texts)
    if not any(char.isalpha() for char in combined):
        return None
    try:
        return langid.classify(combined)[0]
    except Exception as e:
        logging.warning(f"Language detection failed: {e}")
        return None


def compute_stats(lines):
    """
    Page-level statistics from line dicts. This is the only place where the
    statistics and their names are defined; the order here is the order in the output.
    Confidence statistics are None when no line has a usable confidence value.
    """
    scores = [
        float(l["conf"]) for l in lines
        if l.get("conf") is not None and math.isfinite(float(l["conf"]))
    ]

    def stat(func, *args):
        return float(func(scores, *args)) if scores else None

    return {
        "PC": stat(np.mean),
        "PC25": stat(np.percentile, 25),
        "PC50": stat(np.median),
        "PC75": stat(np.percentile, 75),
        "N_LONG_ROWTEXT": sum(1 for l in lines if len(l.get("text") or "") > LONG_ROW_LENGTH),
    }


def add_page_info(page):
    """Adds 'stats' and 'language' to a page dict. Returns the same dict."""
    lines = [line for region in page["regions"] for line in region["lines"]]
    page["stats"] = compute_stats(lines)
    page["language"] = detect_language([l["text"] for l in lines if l.get("text")])
    return page