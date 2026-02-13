# YOLO-E Object Extraction with IoU-based NMS

Extract individual objects from images using YOLO-E segmentation with Non-Maximum Suppression (NMS) based on IoU threshold.

## Features

- **YOLO-E Segmentation**: State-of-the-art object detection and segmentation
- **IoU-based NMS**: Remove overlapping detections with configurable IoU threshold
- **Clean Extraction**: Extract objects with white background masking
- **Flexible Interface**: Command-line script and interactive Jupyter notebook
- **Batch Processing**: Save individual objects or visualization grids

## Quick Start

### Installation

```bash
pip install ultralytics opencv-python numpy pillow matplotlib
```

### Command Line Usage

```bash
# Basic usage
python extract_objects.py --img path/to/image.jpg

# With custom parameters
python extract_objects.py \
    --img path/to/image.jpg \
    --model yolo11l-seg.pt \
    --conf 0.25 \
    --iou 0.5 \
    --padding 20 \
    --gpu 0 \
    --output results \
    --save-individual

# Use specific GPU
python extract_objects.py --img image.jpg --gpu 5
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--img` | str | required | Path to input image |
| `--model` | str | `yolo11l-seg.pt` | YOLO-E model path |
| `--conf` | float | `0.2` | Confidence threshold (0-1) |
| `--iou` | float | `0.5` | IoU threshold for NMS (0-1) |
| `--padding` | int | `20` | Padding around bounding boxes (pixels) |
| `--gpu` | int | `0` | GPU device number |
| `--output` | str | `output` | Output directory |
| `--save-individual` | flag | False | Save each object as separate file |

## Output Structure

```
output/
├── original_image.png              # Original input image
├── extracted_objects_grid.png      # Visualization grid of all objects
└── objects/                        # Individual object images (if --save-individual)
    ├── object_001_conf0.95.png
    ├── object_002_conf0.87.png
    └── ...
```

## How It Works

1. **Detection**: YOLO-E detects objects and generates segmentation masks
2. **NMS Filtering**: Remove overlapping detections based on IoU threshold
   - Sort detections by confidence (descending)
   - Keep highest confidence detection
   - Remove overlapping detections (IoU > threshold)
3. **Extraction**: Extract each object with mask applied
   - Crop to bounding box with padding
   - Apply segmentation mask
   - Set background to white
4. **Output**: Save visualization and/or individual objects

## Examples

### Example 1: Extract objects from image

```bash
python extract_objects.py --img sample.jpg --conf 0.3 --iou 0.5
```

### Example 2: High confidence with aggressive NMS

```bash
python extract_objects.py --img image.jpg --conf 0.5 --iou 0.3 --save-individual
```

### Example 3: Conservative filtering with more padding

```bash
python extract_objects.py --img photo.png --iou 0.7 --padding 30
```

## Requirements

- Python 3.8+
- PyTorch
- ultralytics
- opencv-python
- numpy
- pillow
- matplotlib

## Model Download

On first run, YOLO-E model will be automatically downloaded. Supported models:
- `yolo11n-seg.pt` (nano, fastest)
- `yolo11s-seg.pt` (small)
- `yolo11m-seg.pt` (medium)
- `yolo11l-seg.pt` (large, default)
- `yolo11x-seg.pt` (extra large, most accurate)
