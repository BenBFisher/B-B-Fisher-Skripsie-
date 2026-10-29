import torch
import json
import os
import sys
import io
from PIL import Image
import torchvision.transforms.functional as F_vision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from local_config_rt_detr_v2 import RTDetrV2Config
from local_modeling_rt_detr_v2 import RTDetrV2ForObjectDetection 

def preprocess_image(image_path, target_size=32):
    """Loads, resizes to micro resolution, and normalizes."""
    image = Image.open(image_path).convert("RGB")
    orig_width, orig_height = image.size
    image_resized = image.resize((target_size, target_size))
    img_tensor = F_vision.to_tensor(image_resized)
    img_tensor = F_vision.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return img_tensor.unsqueeze(0), orig_width, orig_height

def post_process_boxes(logits, boxes, orig_width, orig_height, threshold=0.001):
    """Filters boxes and scales coordinates back to the original image dimensions."""
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

def generate_coco_json_from_yolo(img_dir, label_dir, output_json_path, num_classes=2):
    """Dynamically converts YOLO format labels into a COCO ground truth JSON."""
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": f"class_{i}"} for i in range(num_classes)]
    }
    
    ann_id = 1
    valid_images = [f for f in sorted(os.listdir(img_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_id, img_name in enumerate(valid_images):
        img_path = os.path.join(img_dir, img_name)
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
                        if class_id >= num_classes:
                            continue  # Ignore classes outside the 2 target classes
                        x_center, y_center, bbox_width, bbox_height = map(float, parts[1:5])
                        
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

def log_print(msg, file_handle):
    print(msg)
    file_handle.write(msg + "\n")
    file_handle.flush()

def benchmark_accuracy():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_file_path = "benchmark_torch_results_32.txt"
    log_file = open(log_file_path, "w")
    
    log_print(f"--- Initializing PyTorch RT-DETR 32x32 Benchmark on {device.upper()} ---", log_file)
    
    # 1. Update paths to point to your COCO val2017 directory
    val_img_dir = "../phase2_2408_RT_DETR/coco_train2017/images/train"
    val_label_dir = "../phase2_2408_RT_DETR/coco_train2017_small1409_2/labels/train"
    weights_path = "micro_rtdetr_version_4full_epoch_50.pth"
    
    temp_gt_json = "temp_ground_truth.json"
    temp_pred_json = "temp_predictions.json"
    
    if not os.path.exists(val_img_dir) or not os.path.exists(val_label_dir):
        log_print("Error: Could not locate validation images or labels directory.", log_file)
        log_file.close()
        return

    # 2. Match your micro architecture configuration exactly
    config = RTDetrV2Config(
        num_labels=2,
        hidden_dim=32,
        d_model=32,
        encoder_hidden_dim=32,
        decoder_layers=2,
        encoder_in_channels=(24, 48, 96),
        decoder_in_channels=(32, 32, 32),
        encoder_ffn_dim=128,
        decoder_ffn_dim=128
    )
    
    log_print("Converting YOLO annotations to temporary COCO ground truth...", log_file)
    generate_coco_json_from_yolo(val_img_dir, val_label_dir, temp_gt_json, num_classes=config.num_labels)

    log_print(f"Loading weights from {weights_path}...", log_file)
    model = RTDetrV2ForObjectDetection(config).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    model.eval()
    
    coco_gt = COCO(temp_gt_json)
    coco_predictions = []
    img_dicts = coco_gt.loadImgs(coco_gt.getImgIds())
    log_print(f"Running inference across {len(img_dicts)} validation images...", log_file)
    
    with torch.no_grad():
        for count, img_info in enumerate(img_dicts):
            if count > 0 and count % 500 == 0:
                log_print(f"Processed {count}/{len(img_dicts)} frames...", log_file)
                
            img_path = os.path.join(val_img_dir, img_info['file_name'])
            img_id = img_info['id']
            if not os.path.exists(img_path):
                continue
                
            pixel_values, orig_width, orig_height = preprocess_image(img_path, target_size=32)
            pixel_values = pixel_values.to(device)
            
            outputs = model(pixel_values) 
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
        log_print("No predictions generated above threshold.", log_file)
        if os.path.exists(temp_gt_json): os.remove(temp_gt_json)
        log_file.close()
        return

    with open(temp_pred_json, "w") as f:
        json.dump(coco_predictions, f)
        
    log_print("\nRunning official COCO evaluation...", log_file)
    coco_dt = coco_gt.loadRes(temp_pred_json)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    # Intercept summarize output to print and log simultaneously
    stdout_backup = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    coco_eval.summarize()
    sys.stdout = stdout_backup
    
    summary_text = captured_output.getvalue()
    log_print(summary_text, log_file)
    
    if os.path.exists(temp_gt_json): os.remove(temp_gt_json)
    if os.path.exists(temp_pred_json): os.remove(temp_pred_json)
    log_file.close()
    print("Results successfully saved to benchmark_torch_results_32.txt")

if __name__ == "__main__":
    benchmark_accuracy()