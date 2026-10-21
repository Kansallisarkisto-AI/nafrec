# nafrec
An HTR and OCR pipeline for historical text recognition, which classifies text lines into cyrillic or latin based on their script types and then passes them to a proper (TrOCR or PP-OCRv6) text recognition model.

[![ArchXAI project logo](https://centralbaltic.eu/wp-content/uploads/2025/07/ArchXAI_RGB_JPG-335x170.jpg)](https://centralbaltic.eu/project/archxai/)

## Installation

This project uses `pyproject.toml` for dependency management. You can install the required packages using one of the following methods:

### Using uv (recommended)
uv works for all supported platforms including ROCm.
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and activate it
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install correct version of PyTorch for cpu, AMD ROCm or CUDA 13.x
...

# Install package (choose cpu or gpu)
uv pip install git+https://github.com/Kansallisarkisto-AI/nafrec.git[cpu]
```

### Using pip with venv
pip will not work with the ROCm extra.
```bash
# Create a virtual environment and activate it
# Use Python 3.12
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install correct version of PyTorch for cpu, AMD ROCm or CUDA 13.x
...

# Install package (choose cpu or gpu)
pip install git+https://github.com/Kansallisarkisto-AI/nafrec.git[cpu]
```

## Models

The following pre-trained models can be downloaded from Hugging Face:

- Text line and text region detection:
    - **RF-DETR Line and Region Detection Model 2XL**: [https://huggingface.co/Kansallisarkisto/rfdetr-textline-textregion-detection-2xl](https://huggingface.co/Kansallisarkisto/rfdetr-textline-textregion-detection-2xl)
    - **PP-DocLayout_plus-L Region Detection Model**: [https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L_onnx](https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L_onnx)
    - **PP-OCRv6 Medium Line Detection Model**: [https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx)

- Text type classification model:
    - **Cyrillic-latin Text Type Classification Model**: [https://huggingface.co/Kansallisarkisto/cyrillic-latin-textline-classifier](https://huggingface.co/Kansallisarkisto/cyrillic-latin-textline-classifier)

- Cyrillic text recognition:
    - **TrOCR Cyrillic Recognition Model**: 
    [https://huggingface.co/Kansallisarkisto/cyrillic-large-handwritten](https://huggingface.co/Kansallisarkisto/cyrillic-large-handwritten)

- For texts using latin script the model can be chosen based on the language(s) used in the processed documents:
    - **PP-OCRv6 Finnish-Swedish Recognition Model**: 
    [https://huggingface.co/Kansallisarkisto/multicentury-ppocrv6-medium](https://huggingface.co/Kansallisarkisto/multicentury-ppocrv6-medium)
    - **PP-OCRv6 Estonian Recognition Model**: 
    [https://huggingface.co/Kansallisarkisto/estonian-ppocrv6-medium](https://huggingface.co/Kansallisarkisto/estonian-ppocrv6-medium)
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

Most models including the cyrillic model are optional; many different combinations are supported.

### Notes

Nvidia backends ("gpu" and "gpu-optimum") seem stable, but expect small differences if TensorRT is enabled. TensorRT will need to write engine caches into the working directory (in the future we might make this configurable). This backend is currently configured for CUDA 13.x.

CPU backends ("cpu" and "cpu-openvino") seem stable overall.

The AMD GPU execution providers for onnxruntime (ROCm and MIGraphX) are quite unstable. The ROCm provider seems to work on MI250X (only the first call fails, works after retrying), but only with completely static shapes i.e. padding batch size to be static and using the same minimum and maximum width. The MIGraphX provider runs on MI250X but gives incorrect results. MI300+ series and consumer AMD GPUs are untested. Paddle detection and layout models are currently untested on AMD GPUs.

This package is designed to rely on Hugging Face transformers, ONNX Runtime and Roboflow rfdetr libraries only, Paddlepaddle is not a dependency and all pre/postprocessing is handled by this library or our fork of [ppocrv6_onnx](https://github.com/Kansallisarkisto-AI/ppocrv6_onnx). For this reason, the results with Paddle models can differ slightly from Paddlepaddle results.

Our fine-tuned PP-OCRv6 medium models use a raised image height of 96 px compared to the default of 48 px, so they are only guaranteed to work with this package. Most other packages have hardcoded width or height limits that are unsuitable.

#### Choosing package extras
| Extra | Description |
|-------|-------------|
cpu | Basic CPU backend
cpu-openvino | CPU backend with Intel OpenVINO acceleration for ONNX models
gpu | Basic Nvidia GPU backend, also installs correct TensorRT for CUDA 13.x
gpu-optimum | Same as basic Nvidia GPU backend with optimum-onnx acceleration for TrOCR models (xadupre/transformers5 feature branch ahead of release version)
gpu-amd-rocm | Older AMD GPU backend compatible with MI250X (ROCm 7.0.2)
gpu-amd-migraphx | Newer AMD GPU backend, possibly compatible with MI300 series (ROCm 7.2.4)

### Arguments

See [ARGS.md](ARGS.md)

#### Choosing models

- **Latin only, or cyrillic only:** configure just one recognition model. Script classification is disabled and all lines go to that model.
- **Latin and cyrillic:** configure both recognition models (and their processors) plus `--script_classification_model_path`. Each line is classified as `latin` or `cyrillic` and sent to the matching model.
- **PP-OCRv6 for latin:** add `--use_ppocr --ppocr_model_path ...` instead of `--recognition_model_path` / `--processor_path`. The cyrillic model, if used, stays TrOCR.
- **Detection:** the default is RF-DETR. Use `--detector paddle` with `--paddle_det_model_path` (and optionally `--paddle_layout_model_path`) for the Paddle backend.

nafrec checks these combinations at startup and reports missing options before processing begins.

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
    --page_xml 1 \
    --xml_folder /output/path
```

### More examples

Latin-script lines recognized with PP-OCRv6, Paddle detection with a layout model, on two GPUs with fp16 and TensorRT:
```bash
nafrec \
    --input_folder /path/to/images/ \
    --detector paddle \
    --paddle_det_model_path /path/to/det.onnx \
    --paddle_layout_model_path /path/to/layout.onnx \
    --paddle_layout_ignore_classes image chart seal \
    --use_ppocr \
    --ppocr_model_path /path/to/rec.onnx \
    --gpu_ids 0 1 \
    --fp16_det --fp16_rec \
    --enable_tensorrt
```

RF-DETR with tiled detection on CPU only:
```bash
nafrec \
    --input_folder /path/to/images/ \
    --detection_model_path /path/to/rfdetr/model.pth \
    --recognition_model_path /path/to/trocr/model/folder/ \
    --processor_path /path/to/trocr/processor/folder/ \
    --device cpu \
    --cpu_processes 4 \
    --tile_size 768
```

### Python API

nafrec can also be used as a library, for example to keep the models loaded and process images on request, or to supply your own text lines. See [USAGE.md](USAGE.md).

## Output

The pipeline generates XML files (ALTO or PAGE format) containing:
- Detected text region coordinates
- Text line coordinates within each region
- Recognized text for each line
- Model metadata and confidence scores

By default a .json file is also written for each image (disable with `--output_json 0`). It contains for each image
- Image name
- Image height
- Image width
- Page level mean of recognition line confidence values
- Page level median of recognition line confidence values
- Page level 25. percentile of recognition line confidence values
- Page level 75. percentile of recognition line confidence values
- Number of lines where recognition model generated over 100 characters 
- Predicted (majority) language of the recognized text
- Text region detection confidences
- Text region polygons
- Text region names
- For each text region, text line level information containing
    - Recognized text content
    - Text line polygon coordinates
    - Text line polygon detection confidence value
    - Text line recognition confidence value
    - Length of generated text
    - Predicted script type of the row ("latin" or "cyrillic")
    - Script type prediction confidence value