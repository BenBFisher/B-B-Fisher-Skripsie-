import torch
import json
import os
from PIL import Image
import torchvision.transforms.functional as F_vision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Import your configuration and the model architecture
from local_config_rt_detr_v2 import RTDetrV2Config
from local_modeling_rt_detr_v2 import RTDetrV2ForObjectDetection 

def preprocess_image(image_path, target_size=640):
    """Loads, resizes, and normalizes a real image for the model."""
    image = Image.open(image_path).convert("RGB")
    orig_width, orig_height = image.size
    image_resized = image.resize((target_size, target_size))
    img_tensor = F_vision.to_tensor(image_resized)
    img_tensor = F_vision.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return img_tensor.unsqueeze(0), orig_width, orig_height

def post_process_boxes(logits, boxes, orig_width, orig_height, threshold=0.001):
    """Filters boxes and scales coordinates back to the original image."""
    probs = torch.sigmoid(logits[0]) 
    scores, labels = probs.max(dim=-1) 
    
    keep = scores > threshold
    scores = scores[keep]
    labels = labels[keep]
    boxes = boxes[0, keep] 
    
    cx, cy, w, h = boxes.unbind(-1)
    xmin = (cx - 0.5 * w) * orig_width
    ymin = (cy - 0.5 * h) * orig_height
    xmax = (cx + 0.5 * w) * orig_width
    ymax = (cy + 0.5 * h) * orig_height
    
    out_boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
    return scores, labels, out_boxes

def generate_coco_json_from_yolo(img_dir, label_dir, output_json_path, num_classes=80):
    """Dynamically converts a YOLO dataset directory into a COCO JSON file."""
    print("Converting YOLO labels to COCO JSON format...")
    
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": f"class_{i}"} for i in range(num_classes)]
    }
    
    ann_id = 1
    valid_images = [f for f in sorted(os.listdir(img_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_id, img_name in enumerate(valid_images):
        img_path = os.path.join(img_dir, img_name)
        
        # We need exact dimensions to de-normalize YOLO coordinates
        with Image.open(img_path) as img:
            width, height = img.size
            
        coco_format["images"].append({
            "id": img_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })
        
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(label_dir, txt_name)
        
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, bbox_width, bbox_height = map(float, parts[1:5])
                        
                        # Convert YOLO normalized center to COCO absolute top-left
                        abs_w = bbox_width * width
                        abs_h = bbox_height * height
                        abs_xmin = (x_center - (bbox_width / 2)) * width
                        abs_ymin = (y_center - (bbox_height / 2)) * height
                        
                        coco_format["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": class_id,
                            "bbox": [abs_xmin, abs_ymin, abs_w, abs_h],
                            "area": abs_w * abs_h,
                            "iscrowd": 0
                        })
                        ann_id += 1
                        
    with open(output_json_path, "w") as f:
        json.dump(coco_format, f)
    print(f"Generated ground truth JSON with {ann_id - 1} annotations across {len(valid_images)} images.")

def benchmark_accuracy(device="cpu"):
    print(f"--- Initializing YOLO Accuracy Benchmark on {device.upper()} ---")
    
    # 1. File Paths
    # Update these if your YOLO labels are stored in a different folder
    val_img_dir = "./coco128/images/train2017"
    val_label_dir = "./coco128/labels/train2017" 
    
    temp_gt_json = "temp_ground_truth.json"
    temp_pred_json = "temp_predictions.json"
    
    if not os.path.exists(val_img_dir) or not os.path.exists(val_label_dir):
        print(f"Error: Could not find image or label directories.")
        return

    # 2. Convert Ground Truth Data
    config = RTDetrV2Config(num_labels=80) 
    generate_coco_json_from_yolo(val_img_dir, val_label_dir, temp_gt_json, num_classes=config.num_labels)

    # 3. Initialize Architecture and Load Weights
    print("Initializing model architecture...")
    model = RTDetrV2ForObjectDetection(config).to(device)
    
    print("Loading custom trained weights...")
    try:
        model.load_state_dict(torch.load("rt_detr_custom_trained.pth", map_location=torch.device(device)))
    except FileNotFoundError:
        print("Warning: rt_detr_custom_trained.pth not found. Running with random weights (mAP will be 0).")

    """Insert below to use custom weights, and add file name:
    print("Loading custom trained weights...")

    # Put your actual weights file path here
    my_weights_path = "YOUR_WEIGHTS_FILE_NAME_HERE.pth" 
    
    try:
        model.load_state_dict(torch.load(my_weights_path, map_location=torch.device(device)))
    except FileNotFoundError:
        print(f"Warning: {my_weights_path} not found. Running with random weights (mAP will be 0).")"""
    
    model.eval()
    
    # 4. Load COCO Ground Truth
    coco_gt = COCO(temp_gt_json)
    coco_predictions = []
    
    # 5. Run Inference Loop
    img_dicts = coco_gt.loadImgs(coco_gt.getImgIds())
    print(f"Running inference on {len(img_dicts)} images. This may take a moment...")
    
    with torch.no_grad():
        for count, img_info in enumerate(img_dicts):
            if count > 0 and count % 20 == 0:
                print(f"Processed {count}/{len(img_dicts)} images...")
                
            img_path = os.path.join(val_img_dir, img_info['file_name'])
            img_id = img_info['id']
            
            if not os.path.exists(img_path):
                continue
                
            pixel_values, orig_width, orig_height = preprocess_image(img_path)
            pixel_values = pixel_values.to(device)
            
            outputs = model(pixel_values) 
            
            # Use an ultra-low threshold to supply pycocotools with enough boxes for the PR curve
            scores, labels, boxes = post_process_boxes(
                outputs["logits"], 
                outputs["pred_boxes"], 
                orig_width, 
                orig_height, 
                threshold=0.001 
            )
            
            for score, label, box in zip(scores, labels, boxes):
                xmin = box[0].item()
                ymin = box[1].item()
                w = box[2].item() - xmin
                h = box[3].item() - ymin
                
                coco_predictions.append({
                    "image_id": img_id,
                    "category_id": int(label.item()),
                    "bbox": [xmin, ymin, w, h],
                    "score": float(score.item())
                })
                
    if not coco_predictions:
        print("No predictions generated. The model output no boxes above the threshold.")
        if os.path.exists(temp_gt_json): os.remove(temp_gt_json)
        return

    # 6. Save and Evaluate
    print("\nSaving temporary predictions...")
    with open(temp_pred_json, "w") as f:
        json.dump(coco_predictions, f)
        
    print("Running official COCO evaluation...")
    coco_dt = coco_gt.loadRes(temp_pred_json)
    
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # 7. Cleanup Temporary Files
    if os.path.exists(temp_gt_json): os.remove(temp_gt_json)
    if os.path.exists(temp_pred_json): os.remove(temp_pred_json)
        
    print("--- YOLO Accuracy Benchmark Complete ---")

if __name__ == "__main__":
    benchmark_accuracy(device="cpu")