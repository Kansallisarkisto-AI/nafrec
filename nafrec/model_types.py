"""Model types used by the pipeline, and their natural-language descriptions for the XML outputs.

The enum lives here (not in main.py) so xml_output.py can use it without a circular import.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ModelType(Enum):
    PPOCRV6_MEDIUM_REC = 1
    TROCR = 2
    RFDETR = 3
    PPOCRV6_MEDIUM_DET = 4
    PPLAYOUT_L_V1 = 5
    NONE = 6
    CALLER_SUPPLIED_LINES = 7
    LATIN_TROCR_AND_CYRILLIC_TROCR = 8
    LATIN_PPOCRV6_AND_CYRILLIC_TROCR = 9


@dataclass(frozen=True)
class StepDescription:
    """How a processing step is described in the XML.

    creator / software are None when no model was involved (nothing to credit).
    """
    description: str
    creator: Optional[str] = None
    software: Optional[str] = None


# softwareCreator / softwareName values recorded in the XML
_KANSALLISARKISTO = "Kansallisarkisto-AI"
_ROBOFLOW = "Roboflow"
_HUGGINGFACE = "Hugging Face"
_RFDETR = "rfdetr>=1.11.2"
_TRANSFORMERS = "transformers>5"
_PPOCRV6_ONNX = "Kansallisarkisto-AI/ppocrv6_onnx"  # PP-OCRv6 recognition
_NAFREC = "Kansallisarkisto-AI/nafrec"              # Paddle layout / text detection

_NO_MODEL_CALLER = "supplied by the calling application, not detected by NAFRec"

REGION_STEPS = {
    ModelType.RFDETR: StepDescription(
        "Text region detection with an RF-DETR instance segmentation model",
        _ROBOFLOW, _RFDETR),
    ModelType.PPLAYOUT_L_V1: StepDescription(
        "Text region detection with the PP-DocLayout-L layout analysis model",
        _KANSALLISARKISTO, _NAFREC),
    ModelType.NONE: StepDescription(
        "No region detection model was used, so the whole page is treated as a single text region"),
    ModelType.CALLER_SUPPLIED_LINES: StepDescription(
        f"Text regions were {_NO_MODEL_CALLER}"),
}

LINE_STEPS = {
    ModelType.RFDETR: StepDescription(
        "Text line detection with an RF-DETR instance segmentation model",
        _ROBOFLOW, _RFDETR),
    ModelType.PPOCRV6_MEDIUM_DET: StepDescription(
        "Text line detection with the PP-OCRv6 medium text detection model",
        _KANSALLISARKISTO, _NAFREC),
    ModelType.NONE: StepDescription(
        "No text line detection model was used"),
    ModelType.CALLER_SUPPLIED_LINES: StepDescription(
        f"Text lines were {_NO_MODEL_CALLER}"),
}

RECOGNITION_STEPS = {
    ModelType.TROCR: StepDescription(
        "Handwritten text recognition with a TrOCR model",
        _HUGGINGFACE, _TRANSFORMERS),
    ModelType.PPOCRV6_MEDIUM_REC: StepDescription(
        "Text recognition with the PP-OCRv6 medium text recognition model",
        _KANSALLISARKISTO, _PPOCRV6_ONNX),
    ModelType.LATIN_TROCR_AND_CYRILLIC_TROCR: StepDescription(
        "Handwritten text recognition with two TrOCR models: one for latin-script "
        "text lines and one for cyrillic-script text lines, chosen per line by script classification",
        _HUGGINGFACE, _TRANSFORMERS),
    ModelType.LATIN_PPOCRV6_AND_CYRILLIC_TROCR: StepDescription(
        "Text recognition with two models: the PP-OCRv6 medium text recognition model for "
        "latin-script text lines and a TrOCR model for cyrillic-script text lines, "
        "chosen per line by script classification",
        f"{_KANSALLISARKISTO} and {_HUGGINGFACE}",
        f"{_PPOCRV6_ONNX} and {_TRANSFORMERS}"),
}


def describe_step(steps, model_type, step_name):
    """Look up a step description; a type that doesn't belong to the step is a programming error."""
    try:
        return steps[model_type]
    except KeyError:
        raise ValueError(
            f"{model_type!r} is not a valid model type for {step_name}") from None


def combine_recognition_types(latin_type=None, cyrillic_type=None):
    """Model type describing the recognizers used on a page.

    latin_type / cyrillic_type: ModelType of the recognizer that transcribed at least one
    line of that script on the page, or None if it transcribed none.
    A single recognizer keeps its own type; latin + cyrillic give the combined types.
    """
    if latin_type is None and cyrillic_type is None:
        return ModelType.NONE
    if cyrillic_type is not None and cyrillic_type != ModelType.TROCR:
        raise ValueError(
            f"Only TrOCR is supported for cyrillic recognition, got {cyrillic_type!r}")
    if latin_type is None:
        return cyrillic_type
    if cyrillic_type is None:
        return latin_type
    if latin_type == ModelType.TROCR:
        return ModelType.LATIN_TROCR_AND_CYRILLIC_TROCR
    if latin_type == ModelType.PPOCRV6_MEDIUM_REC:
        return ModelType.LATIN_PPOCRV6_AND_CYRILLIC_TROCR
    raise ValueError(f"Unsupported latin recognition type {latin_type!r}")
