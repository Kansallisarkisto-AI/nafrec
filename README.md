# nafrec
An HTR and OCR pipeline for historical text recognition, which classifies text lines into cyrillic or latin based on their script types and then passes them to a proper (TrOCR or PP-OCRv6) text recognition model.

## Installation

This project uses `pyproject.toml` for dependency management. You can install the required packages using one of the following methods:

### Using uv (recommended)
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and activate it
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package (choose cpu or gpu)
uv pip install nafrec[cpu]
```

### Using pip with venv
```bash
# Create a virtual environment and activate it
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package (choose cpu or gpu)
pip install nafrec[cpu]
```

## Models

The following pre-trained models can be downloaded from Hugging Face:

- Text line and text region detection:
    - **RF-DETR Detection Model**: [https://huggingface.co/Kansallisarkisto/rfdetr_textline_textregion_detection_model](https://huggingface.co/Kansallisarkisto/rfdetr_textline_textregion_detection_model)

- Text type classification model:
    - **Cyrillic-latin Text Type Classification Model**: [https://huggingface.co/Kansallisarkisto/cyrillic-latin-textline-classifier](https://huggingface.co/Kansallisarkisto/cyrillic-latin-textline-classifier)

- Cyrillic text recognition:
    - **TrOCR Cyrillic Recognition Model**: 
    [https://huggingface.co/Kansallisarkisto/cyrillic-large-handwritten](https://huggingface.co/Kansallisarkisto/cyrillic-large-handwritten)

- For texts using latin script the model can be chosen based on the language(s) used in the processed documents:
    - **TrOCR Finnish-Swedish Recognition Model**: 
    [https://huggingface.co/Kansallisarkisto/multicentury-htr-model](https://huggingface.co/Kansallisarkisto/multicentury-htr-model)
    - **TrOCR Estonian Recognition Model**: 
    [https://huggingface.co/Kansallisarkisto/estonian-large-handwritten](https://huggingface.co/Kansallisarkisto/estonian-large-handwritten)
    - **TrOCR Latvian Recognition Model**: 
    [https://huggingface.co/Kansallisarkisto/latvian-large-handwritten](https://huggingface.co/Kansallisarkisto/latvian-large-handwritten)

After downloading, update the model paths in your command line arguments or configuration.

## Pipeline Overview

The pipeline processes images through the following steps:

1. **Input**: Historical document image
2. **Detection**: RF-DETR model detects text regions and text lines 
3. **Cropping**: Text lines are cropped from the original image based on detected coordinates
4. **Text type classification**: Classification model labels every text line as "cyrillic" or "latin" based on the predicted script type
5. **Recognition**: Based on the classification results, text line images ase passed to the proper TrOCR or PP-OCRv6 model which recognizes text from each cropped line
6. **Output**: ALTO XML and/or PAGE XML file containing region coordinates, text line coordinates, and recognized text

## Usage

Run the pipeline using the command `nafrec`:
```bash
nafrec \
    --detection_model_path /path/to/rfdetr/model.pth \
    --script_classification_model_path /path/to/script/classification/model.pt \
    --recognition_model_path /path/to/trocr/model/folder/ \
    --cyrillic_recognition_model_path /path/to/cyrillic/trocr/model/folder/ \
    --processor_path /path/to/trocr/processor/folder/ \
    --cyrillic_processor_path /path/to/cyrillic/trocr/processor/folder/ \
    --input_folder /path/to/images/
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--detection_model_path` | str | `/path/to/rfdetr/model.pth` | Path to the RF-DETR detection model file |
| `--script_classification_model_path` | str | `/path/to/script/classification/model.pt` | Path to the script classification model file |
| `--recognition_model_path` | str | `/path/to/trocr/model/folder/` | Path to the TrOCR recognition model folder |
| `--cyrillic_recognition_model_path` | str | `/path/to/cyrillic/trocr/model/folder/` | Path to the cyrillic TrOCR recognition model folder |
| `--processor_path` | str | `/path/to/trocr/processor/folder/` | Path to the TrOCR processor folder |
| `--cyrillic_processor_path` | str | `/path/to/trocr/processor/folder/` | Path to the cyrillic TrOCR processor folder |
| `--input_folder` | str | **required** | Path to folder containing input images |
| `--region_model_name` | str | `rfdetr_text_seg_model_202510` | Region detection model name |
| `--line_model_name` | str | `rfdetr_text_seg_model_202510` | Line detection model name |
| `--script_classification_model_name` | str | `script_classifier_202610` | Script classification model name |
| `--text_rec_model_name` | str | `202509_tf32` | Text recognition model name |
| `--cyrillic_text_rec_model_name` | str | `cyrillic_large_202603` | Cyrillic text recognition model name |
| `--trocr_batch_size` | int | 8 | Batch size for text recognition |
| `--classifier_batch_size` | int | 32 | Batch size for script type classification |
| `--classifier_default_label` | str | latin |Default label used by script type classifier in cases where all line classifications fail |
| `--page_xml` | int | 0 | Whether to save output as PAGE XML (0 = no, 1 = yes) |
| `--alto_xml` | int | 1 | Whether to save output as ALTO XML (0 = no, 1 = yes) |
| `--output_json` | int | 0 | Whether to save output as .json file (0 = no, 1 = yes) |
| `--xml_folder` | str | None | Custom path for XML output. If None, saves to `input_folder/alto` or `input_folder/page` |
| `--json_folder` | str | None | Custom path for .json output. If None, saves to `input_folder/json` |
| `--confidence_threshold` | float | 0.15 | Detection confidence threshold for filtering detections |
| `--line_percentage_threshold` | float | 7e-05 | Threshold value for filtering out small line polygons |
| `--region_percentage_threshold` | float | 7e-05 | Threshold value for filtering out small region polygons |
| `--line_iou` | float | 0.3 | Threshold value for merging lines based on intersection over union (IoU) |
| `--region_iou` | float | 0.3 | Threshold value for merging regions based on intersection over union (IoU) |
| `--line_overlap_threshold` | float | 0.5 | Threshold value for merging lines based on overlapping area |
| `--region_overlap_threshold` | float | 0.5 | Threshold value for merging regions based on overlapping area |

### Example
```bash
nafrec \
    --detection_model_path /path/to/rfdetr/model.pth \
    --script_classification_model_path /path/to/script/classification/model.pt \
    --recognition_model_path /path/to/trocr/model/folder/ \
    --cyrillic_recognition_model_path /path/to/cyrillic/trocr/model/folder/ \
    --processor_path /path/to/trocr/processor/folder/ \
    --cyrillic_processor_path /path/to/cyrillic/trocr/processor/folder/ \
    --input_folder /path/to/images/ \
    --confidence_threshold 0.2 \
    --page_xml True \
    --xml_folder /output/path
```

## Output

The pipeline generates XML files (ALTO or PAGE format) containing:
- Detected text region coordinates
- Text line coordinates within each region
- Recognized text for each line
- Model metadata and confidence scores

Optionally also output files in .json format can be generated. They contain for each image
- Image name
- Image height
- Image width
- Page level mean of TrOCR row confidence values
- Page level median of TrOCR row confidence values
- Page level 25. percentile of TrOCR row confidence values
- Page level 75. percentile of TrOCR row confidence values
- Number of rows where TrOCR model generated over 100 characters 
- Predicted (majority) language of the recognized text
- Text region detection confidences
- Text region polygons
- Text region names
- For each text region, text line level information containing
    - Recognized text content
    - Text line polygon coordinates
    - Text line polygon detection confidence value
    - Text recognition confidence value
    - Length of generated text
    - Predicted script type of the row ("latin" or "cyrillic")
    - Script type prediction confidence value