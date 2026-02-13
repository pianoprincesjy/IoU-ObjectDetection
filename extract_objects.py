"""
YOLO-E Object Extraction with IoU-based NMS Filtering

This script uses YOLO-E segmentation model to detect and extract individual objects
from an image with Non-Maximum Suppression (NMS) based on IoU threshold.
"""

import os
import argparse
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from ultralytics import YOLOE
except ImportError:
    print("Error: ultralytics package not found. Install with: pip install ultralytics")
    exit(1)


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)
    
    Returns:
        float: IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection area
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


def extract_objects_with_nms(results, original_image, padding=10, iou_threshold=0.5):
    """
    Extract segmented objects with NMS filtering.
    
    Args:
        results: YOLO prediction results
        original_image: PIL Image object
        padding: Padding around bounding boxes (pixels)
        iou_threshold: IoU threshold for NMS (0-1)
    
    Returns:
        extracted_objects: List of extracted object images (PIL)
        object_info: List of object information dicts
    """
    
    if not (hasattr(results[0], 'masks') and results[0].masks is not None):
        print("No segmentation masks found in results.")
        return [], []
    
    img_array = np.array(original_image)
    
    # Convert RGBA to RGB if necessary
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    
    masks = results[0].masks.data.cpu().numpy()
    boxes = results[0].boxes.xyxy.cpu().numpy()
    conf = results[0].boxes.conf.cpu().numpy()
    cls = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else np.zeros(len(boxes))
    
    # Sort by confidence (descending)
    sorted_indices = np.argsort(conf)[::-1]
    
    # Apply NMS
    keep_indices = []
    for i in sorted_indices:
        current_box = boxes[i]
        
        should_keep = True
        for kept_idx in keep_indices:
            kept_box = boxes[kept_idx]
            iou = calculate_iou(current_box, kept_box)
            
            if iou > iou_threshold:
                should_keep = False
                break
        
        if should_keep:
            keep_indices.append(i)
    
    print(f"Before NMS: {len(boxes)} objects")
    print(f"After NMS: {len(keep_indices)} objects (IoU threshold: {iou_threshold})")
    print(f"Removed: {len(boxes) - len(keep_indices)} overlapping objects\n")
    
    # Extract selected objects
    extracted_objects = []
    object_info = []
    
    for idx, original_idx in enumerate(keep_indices):
        mask = masks[original_idx]
        box = boxes[original_idx]
        confidence = conf[original_idx]
        class_id = cls[original_idx]
        
        # Resize mask to image size
        h, w = img_array.shape[:2]
        resized_mask = cv2.resize(mask, (w, h))
        
        # Apply padding to bounding box
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Crop image and mask
        cropped_img = img_array[y1:y2, x1:x2]
        cropped_mask = resized_mask[y1:y2, x1:x2]
        
        # Apply mask (background to white)
        masked_obj = cropped_img.copy()
        
        n_channels = masked_obj.shape[-1]
        if len(cropped_mask.shape) == 2:
            cropped_mask_expanded = np.stack([cropped_mask] * n_channels, axis=-1)
        else:
            cropped_mask_expanded = cropped_mask
            
        masked_obj[cropped_mask_expanded < 0.5] = 255
        
        # Convert to PIL Image
        pil_obj = Image.fromarray(masked_obj.astype(np.uint8))
        
        extracted_objects.append(pil_obj)
        object_info.append({
            'index': idx + 1,
            'original_index': original_idx + 1,
            'class_id': int(class_id),
            'confidence': float(confidence),
            'bbox': (x1, y1, x2, y2),
            'size': pil_obj.size
        })
    
    return extracted_objects, object_info


def visualize_results(original_image, extracted_objects, object_info, output_dir):
    """
    Visualize and save extraction results.
    
    Args:
        original_image: PIL Image
        extracted_objects: List of extracted object images
        object_info: List of object information
        output_dir: Output directory path
    """
    
    if not extracted_objects:
        print("No objects to visualize.")
        return
    
    n_objects = len(extracted_objects)
    cols = min(4, n_objects)
    rows = (n_objects + cols - 1) // cols
    
    fig = plt.figure(figsize=(16, 4 * rows))
    
    for i in range(n_objects):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(extracted_objects[i])
        plt.title(f"Object {object_info[i]['index']}\n"
                 f"Confidence: {object_info[i]['confidence']:.2f}",
                 fontsize=10)
        plt.axis('off')
    
    plt.suptitle(f'Extracted Objects ({n_objects} total)', fontsize=14, y=0.995)
    plt.tight_layout()
    
    # Save visualization
    viz_path = output_dir / 'extracted_objects_grid.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {viz_path}")
    plt.close()
    
    # Save original image
    orig_path = output_dir / 'original_image.png'
    original_image.save(orig_path)
    print(f"Saved original image: {orig_path}")


def save_individual_objects(extracted_objects, object_info, output_dir):
    """
    Save each extracted object as a separate image file.
    
    Args:
        extracted_objects: List of extracted object images
        object_info: List of object information
        output_dir: Output directory path
    """
    
    objects_dir = output_dir / 'objects'
    objects_dir.mkdir(exist_ok=True)
    
    for obj, info in zip(extracted_objects, object_info):
        filename = f"object_{info['index']:03d}_conf{info['confidence']:.2f}.png"
        filepath = objects_dir / filename
        obj.save(filepath)
    
    print(f"Saved {len(extracted_objects)} individual objects to: {objects_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract objects from images using YOLO-E with IoU-based NMS filtering'
    )
    parser.add_argument('--img', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model', type=str, default='yolo11l-seg.pt',
                       help='YOLO-E model path (default: yolo11l-seg.pt)')
    parser.add_argument('--conf', type=float, default=0.2,
                       help='Confidence threshold (default: 0.2)')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU threshold for NMS (default: 0.5)')
    parser.add_argument('--padding', type=int, default=20,
                       help='Padding around bounding boxes (default: 20)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device number (default: 0)')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory (default: output)')
    parser.add_argument('--save-individual', action='store_true',
                       help='Save each object as individual image file')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Check input image
    if not os.path.exists(args.img):
        print(f"Error: Image not found: {args.img}")
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("YOLO-E Object Extraction with IoU-based NMS")
    print("=" * 70)
    print(f"Input image: {args.img}")
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"GPU device: {args.gpu}")
    print(f"Output directory: {args.output}")
    print("=" * 70)
    print()
    
    # Load model
    print("Loading YOLO-E model...")
    try:
        model = YOLOE(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model file exists or will be downloaded automatically.")
        return
    
    # Load image
    print(f"Loading image: {args.img}")
    image = Image.open(args.img).convert('RGB')
    print(f"Image size: {image.size}\n")
    
    # Run prediction
    print("Running YOLO-E segmentation...")
    results = model.predict(image, conf=args.conf, verbose=False)
    
    if not results or not results[0].boxes:
        print("No objects detected in the image.")
        return
    
    print(f"Detected {len(results[0].boxes)} objects\n")
    
    # Extract objects with NMS
    print("Extracting objects with NMS filtering...")
    extracted_objects, object_info = extract_objects_with_nms(
        results, image, 
        padding=args.padding, 
        iou_threshold=args.iou
    )
    
    if not extracted_objects:
        print("No objects extracted after NMS filtering.")
        return
    
    # Print object information
    print("Extracted objects:")
    for info in object_info:
        print(f"  Object {info['index']}: "
              f"confidence={info['confidence']:.3f}, "
              f"size={info['size']}, "
              f"bbox={info['bbox']}")
    print()
    
    # Visualize and save results
    print("Creating visualizations...")
    visualize_results(image, extracted_objects, object_info, output_dir)
    
    # Save individual objects if requested
    if args.save_individual:
        print("\nSaving individual objects...")
        save_individual_objects(extracted_objects, object_info, output_dir)
    
    print("\n" + "=" * 70)
    print("Extraction completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
