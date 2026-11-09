"""
XML output for nafrec: writes a page dict (format: see alto_writer.py) as ALTO and/or
PAGE XML with the shared writers (alto_writer.py, page_writer.py).

The processing metadata (descriptions, software and creators per model type) comes
from model_types.py; this module only turns it into ProcessingSteps.
"""
import os
from pathlib import Path

from .alto_writer import ProcessingStep, write_alto
from .page_writer import write_page_xml
from .model_types import REGION_STEPS, LINE_STEPS, RECOGNITION_STEPS, describe_step

# Script classification step (only written when a classification model was used)
SCRIPT_CLASSIFICATION_DESCRIPTION = "Text line script classification"
SCRIPT_CLASSIFICATION_SOFTWARE = "MobileNet_V3_Small"
SCRIPT_CLASSIFICATION_CREATOR = "Pytorch"


def _processing_step(step_id, step, model_name):
    """ProcessingStep from a model_types.StepDescription. Software (with the model name as
    its version) is only credited when a model was used in the step."""
    if step.software:
        return ProcessingStep(step_id, step.description, step.software, model_name, step.creator)
    return ProcessingStep(step_id, step.description)


def get_processing_steps(xml_input):
    """
    Processing metadata for the ALTO Description, from the model names and types in
    xml_input (a main.XmlInput). The step with id TEXT_RECOGNITION also carries the
    page statistics.
    """
    steps = [
        _processing_step("TEXT_REGION_SEGMENTATION",
                         describe_step(REGION_STEPS, xml_input.region_segment_model_type, "region detection"),
                         xml_input.region_segment_model_name),
        _processing_step("TEXT_LINE_SEGMENTATION",
                         describe_step(LINE_STEPS, xml_input.line_segment_model_type, "line detection"),
                         xml_input.line_segment_model_name),
    ]
    if xml_input.classification_model_name:
        steps.append(ProcessingStep("SCRIPT_CLASSIFICATION", SCRIPT_CLASSIFICATION_DESCRIPTION,
                                    SCRIPT_CLASSIFICATION_SOFTWARE, xml_input.classification_model_name,
                                    SCRIPT_CLASSIFICATION_CREATOR))
    steps.append(_processing_step("TEXT_RECOGNITION",
                                  describe_step(RECOGNITION_STEPS, xml_input.text_recognition_model_type,
                                                "text recognition"),
                                  xml_input.text_recognition_model_name))
    return steps


def _step_text(step, model_name):
    """Natural-language one-liner for a processing step, e.g. for PAGE XML <Comments>."""
    text = step.description
    if step.software and model_name:
        text += f" ({model_name})"
    return text


def describe_processing(xml_input):
    """Natural-language summary of the models used, for the PAGE XML Metadata/Comments."""
    parts = [
        _step_text(describe_step(REGION_STEPS, xml_input.region_segment_model_type,
                                 "region detection"), xml_input.region_segment_model_name),
        _step_text(describe_step(LINE_STEPS, xml_input.line_segment_model_type,
                                 "line detection"), xml_input.line_segment_model_name),
    ]
    if xml_input.classification_model_name:
        parts.append("Text line script classification (latin or cyrillic) "
                     f"({xml_input.classification_model_name})")
    parts.append(_step_text(describe_step(
        RECOGNITION_STEPS, xml_input.text_recognition_model_type, "text recognition"),
        xml_input.text_recognition_model_name))
    return "; ".join(parts)


def save_xml(page, xml_input):
    """
    Saves a page dict as PAGE and/or ALTO XML.

    Args:
        page: page dict (format: see alto_writer.py), as returned by
            main.predict_single_image / NAFRec.process / NAFRec.recognize.
        xml_input: main.XmlInput with the output settings and the models used.

    Writes <xml_path>/page/<image stem>.xml and/or <xml_path>/alto/<image stem>.xml,
    depending on page_xml and alto_xml. Line-level script classification is written to
    ALTO when a classification model was used.
    """
    if not xml_input.xml_path:
        return
    xml_name = Path(xml_input.image_path).stem + '.xml'
    # PAGE XML option
    if xml_input.page_xml:
        save_folder = Path(xml_input.xml_path, 'page')
        os.makedirs(save_folder, exist_ok=True)
        write_page_xml(page, str(Path(save_folder, xml_name)), comments=describe_processing(xml_input))
    # ALTO XML option
    if xml_input.alto_xml:
        save_folder = Path(xml_input.xml_path, 'alto')
        os.makedirs(save_folder, exist_ok=True)
        write_alto(page, get_processing_steps(xml_input), str(Path(save_folder, xml_name)),
                   include_script=bool(xml_input.classification_model_name))