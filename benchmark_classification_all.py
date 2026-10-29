import os
import sys
import json
import argparse
import time
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F_vision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def log_print(msg, file_handle):
    print(msg)
    file_handle.write(msg + "\n")
    file_handle.flush()

def preprocess_image(image_path, target_size):
    image = Image.open(image_path).convert("RGB")
    orig_width, orig_height = image.size
    image_resized = image.resize((target_size, target_size))
    img_tensor = F_vision.to_tensor(image_resized)
    img_tensor = F_vision.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return img_tensor, orig_width, orig_height

def generate_coco_json_from_yolo(img_dir, label_dir, output_json_path, num_classes=80, class_mapping=None):
    if class_mapping is None:
        class_mapping = {}

    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": f"class_{i}"} for i in range(num_classes)]
    }
    ann_id = 1
    valid_images = [f for f in sorted(os.listdir(img_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    gt_dict = {} 
    
    for img_id, img_name in enumerate(valid_images):
        img_path = os.path.join(img_dir, img_name)
        with Image.open(img_path) as img:
            width, height = img.size
            
        coco_format["images"].append({"id": img_id, "file_name": img_name, "width": width, "height": height})
        
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(label_dir, txt_name)
        
        gt_boxes = []
        gt_labels = []
        
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        raw_class_id = int(parts[0])
                        class_id = class_mapping.get(raw_class_id, raw_class_id)
                        
                        x_center, y_center, bbox_width, bbox_height = map(float, parts[1:5])
                        
                        abs_w = bbox_width * width
                        abs_h = bbox_height * height
                        abs_xmin = (x_center - (bbox_width / 2.0)) * width
                        abs_ymin = (y_center - (bbox_height / 2.0)) * height
                        abs_xmax = abs_xmin + abs_w
                        abs_ymax = abs_ymin + abs_h
                        
                        coco_format["annotations"].append({
                            "id": ann_id, 
                            "image_id": img_id, 
                            "category_id": class_id,
                            "bbox": [abs_xmin, abs_ymin, abs_w, abs_h], 
                            "area": abs_w * abs_h, 
                            "iscrowd": 0
                        })
                        ann_id += 1
                        
                        gt_boxes.append([abs_xmin, abs_ymin, abs_xmax, abs_ymax])
                        gt_labels.append(class_id)
                        
        gt_dict[img_id] = {"boxes": gt_boxes, "labels": gt_labels}
                        
    with open(output_json_path, "w") as f:
        json.dump(coco_format, f)
    return gt_dict

def evaluate_containment(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
    hits = 0
    total_gt = len(gt_boxes)
    
    hit_confidences = []
    miss_confidences = []

    for p_box, p_label, p_score in zip(pred_boxes, pred_labels, pred_scores):
        cx = (p_box[0] + p_box[2]) / 2.0
        cy = (p_box[1] + p_box[3]) / 2.0
        is_tp = False
        
        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            if p_label == gt_label and (gt_box[0] <= cx <= gt_box[2]) and (gt_box[1] <= cy <= gt_box[3]):
                is_tp = True
                break
                
        if is_tp:
            hit_confidences.append(p_score)
        else:
            miss_confidences.append(p_score)
            
    if total_gt > 0:
        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            hit_found = False
            for p_box, p_label in zip(pred_boxes, pred_labels):
                if p_label != gt_label: 
                    continue
                cx = (p_box[0] + p_box[2]) / 2.0
                cy = (p_box[1] + p_box[3]) / 2.0
                if (gt_box[0] <= cx <= gt_box[2]) and (gt_box[1] <= cy <= gt_box[3]):
                    hit_found = True
                    break 
            if hit_found: 
                hits += 1
            
    return hits, total_gt, hit_confidences, miss_confidences

def main():
    parser = argparse.ArgumentParser(description="Multi-Threshold RT-DETR Universal Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Path to .onnx, .tflite, .pt, .pth, safetensors, or HF directory")
    parser.add_argument("--size", type=int, default=32, help="Input resolution (e.g. 32, 160, 640)")
    parser.add_argument("--val_images", type=str, required=True, help="Path to validation image directory")
    parser.add_argument("--val_labels", type=str, required=True, help="Path to YOLO txt directory")
    parser.add_argument("--num_classes", type=int, default=2, help="Total number of classes")
    parser.add_argument("--map_class_0", type=int, default=0, help="Class ID to map ground truth 0 to")
    parser.add_argument("--map_class_1", type=int, default=1, help="Class ID to map ground truth 1 to")
    args = parser.parse_args()

    model_ext = os.path.splitext(args.model)[1].lower()
    model_basename = os.path.basename(os.path.normpath(args.model))
    log_file_path = f"benchmark_comprehensive_{model_basename}_{args.size}.txt"
    log_file = open(log_file_path, "w")
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    log_print("--- Initializing Multi-Threshold Comprehensive Benchmark ---", log_file)
    
    temp_gt_json = "temp_ground_truth.json"
    temp_pred_json = "temp_predictions.json"
    
    custom_mapping = {0: args.map_class_0, 1: args.map_class_1}
    
    gt_dict = generate_coco_json_from_yolo(
        args.val_images, 
        args.val_labels, 
        temp_gt_json, 
        num_classes=args.num_classes, 
        class_mapping=custom_mapping
    )
    
    coco_gt = COCO(temp_gt_json)
    img_dicts = coco_gt.loadImgs(coco_gt.getImgIds())
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # -------------------------------------------------------------
    # 1. Hugging Face Models (.safetensors or model directory)
    # -------------------------------------------------------------
    if os.path.isdir(args.model) or model_ext == ".safetensors":
        from transformers import RTDetrV2ForObjectDetection
        model_dir = args.model if os.path.isdir(args.model) else os.path.dirname(args.model)
        log_print(f"Loading Hugging Face model from: {model_dir}", log_file)
        model = RTDetrV2ForObjectDetection.from_pretrained(model_dir, local_files_only=True).to(device)
        model.eval()
        model_ext = ".hf"

    # -------------------------------------------------------------
    # 2. ONNX Runtime
    # -------------------------------------------------------------
    elif model_ext == ".onnx":
        import onnxruntime as ort
        session = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])

    # -------------------------------------------------------------
    # 3. TensorFlow Lite (Float32, INT8, UINT8)
    # -------------------------------------------------------------
    elif model_ext == ".tflite":
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=args.model)
        interpreter.allocate_tensors()

    # -------------------------------------------------------------
    # 4. Native PyTorch Checkpoint (.pt / .pth)
    # -------------------------------------------------------------
    elif model_ext in [".pt", ".pth"]:
        checkpoint = torch.load(args.model, map_location=torch.device(device), weights_only=False)
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint

        # Detect Backbone Architecture from projection layer input channels
        is_resnet = False
        in_channels = (24, 48, 96)  # Default MobileNet
        
        if "encoder_input_proj.0.0.weight" in state_dict:
            first_proj_in_ch = state_dict["encoder_input_proj.0.0.weight"].shape[1]
            if first_proj_in_ch == 128:
                is_resnet = True
                in_channels = (128, 256, 512)
            elif first_proj_in_ch == 32:
                in_channels = (32, 64, 128)
            elif first_proj_in_ch == 24:
                in_channels = (24, 48, 96)

        target_folder = os.path.abspath("./3107_RT_DETR") if is_resnet else os.path.abspath("./phase3_0410_RT_DETR")
        if target_folder not in sys.path:
            sys.path.insert(0, target_folder)

        from local_config_rt_detr_v2 import RTDetrV2Config
        from local_modeling_rt_detr_v2 import RTDetrV2ForObjectDetection

        num_classes_found = args.num_classes
        hidden_dim_found = 256 if is_resnet else 32
        
        for k, v in state_dict.items():
            if "class_embed" in k and "weight" in k and len(v.shape) == 2:
                num_classes_found = v.shape[0]
                hidden_dim_found = v.shape[1]
                break

        print(f"Detected Checkpoint Spec: Backbone={'ResNet-18' if is_resnet else 'MobileNet'} | Classes={num_classes_found} | HiddenDim={hidden_dim_found}")

        if is_resnet:
            config = RTDetrV2Config(
                num_labels=num_classes_found,
                hidden_dim=hidden_dim_found,
                d_model=hidden_dim_found,
                encoder_hidden_dim=hidden_dim_found,
                decoder_layers=6,
                encoder_in_channels=(128, 256, 512),
                decoder_in_channels=(hidden_dim_found, hidden_dim_found, hidden_dim_found),
                encoder_ffn_dim=1024,
                decoder_ffn_dim=1024
            )
        elif num_classes_found == 2 and hidden_dim_found == 32:
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
        else:
            config = RTDetrV2Config(
                num_labels=num_classes_found,
                hidden_dim=hidden_dim_found,
                d_model=hidden_dim_found,
                encoder_hidden_dim=hidden_dim_found,
                decoder_layers=6,
                encoder_in_channels=in_channels,
                decoder_in_channels=(hidden_dim_found, hidden_dim_found, hidden_dim_found),
                encoder_ffn_dim=1024,
                decoder_ffn_dim=1024
            )
            
        model = RTDetrV2ForObjectDetection(config).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
    else:
        raise ValueError(f"Unsupported model format: {model_ext}")
        
    coco_predictions = []
    containment_tracker = {t: {"hits": 0, "gt": 0, "tp_confs": [], "fp_confs": []} for t in thresholds}
    inference_latencies = []
    
    log_print(f"Evaluating {len(img_dicts)} images...", log_file)
    
    for count, img_info in enumerate(img_dicts):
        img_path = os.path.join(args.val_images, img_info['file_name'])
        img_id = img_info['id']
        
        img_tensor, orig_w, orig_h = preprocess_image(img_path, args.size)
        
        t0 = time.perf_counter()
        
        if model_ext == ".onnx":
            img_np = img_tensor.unsqueeze(0).numpy()
            outputs = session.run(None, {session.get_inputs()[0].name: img_np})
            logits = torch.tensor(outputs[0])
            pred_boxes = torch.tensor(outputs[1])
            
        elif model_ext == ".tflite":
            img_np = img_tensor.unsqueeze(0).numpy()
            img_np = np.transpose(img_np, (0, 2, 3, 1))
            
            input_details = interpreter.get_input_details()[0]
            if input_details['dtype'] == np.int8:
                scale, zero_point = input_details['quantization']
                if scale > 0:
                    img_np = (img_np / scale) + zero_point
                img_np = np.clip(img_np, -128, 127).astype(np.int8)
            elif input_details['dtype'] == np.uint8:
                scale, zero_point = input_details['quantization']
                if scale > 0:
                    img_np = (img_np / scale) + zero_point
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.float32)

            interpreter.set_tensor(input_details['index'], img_np)
            interpreter.invoke()
            out_details = interpreter.get_output_details()
            
            logits_tensor = interpreter.get_tensor(out_details[0]['index'])
            boxes_tensor = interpreter.get_tensor(out_details[1]['index'])
            
            if out_details[0]['dtype'] in [np.int8, np.uint8]:
                l_scale, l_zp = out_details[0]['quantization']
                logits_tensor = (logits_tensor.astype(np.float32) - l_zp) * l_scale
            if out_details[1]['dtype'] in [np.int8, np.uint8]:
                b_scale, b_zp = out_details[1]['quantization']
                boxes_tensor = (boxes_tensor.astype(np.float32) - b_zp) * b_scale
                
            logits = torch.tensor(logits_tensor)
            pred_boxes = torch.tensor(boxes_tensor)
            
        elif model_ext in [".pt", ".pth", ".hf"]:
            img_device = img_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_device)
            if hasattr(outputs, "logits"):
                logits = outputs.logits.cpu()
                pred_boxes = outputs.pred_boxes.cpu()
            elif isinstance(outputs, dict):
                logits = outputs["logits"].cpu()
                pred_boxes = outputs["pred_boxes"].cpu()
            else:
                logits = outputs[0].cpu()
                pred_boxes = outputs[1].cpu()
            
        t1 = time.perf_counter()
        
        latency_ms = (t1 - t0) * 1000.0
        inference_latencies.append(latency_ms)
            
        probs = torch.sigmoid(logits[0])
        all_scores, all_labels = probs.max(dim=-1)
        
        keep_coco = all_scores > 0.001 
        c_scores, c_labels, c_boxes = all_scores[keep_coco], all_labels[keep_coco], pred_boxes[0, keep_coco]
        
        for score, label, box in zip(c_scores, c_labels, c_boxes):
            cx, cy, w, h = box.unbind(-1)
            xmin = (cx - 0.5 * w) * orig_w
            ymin = (cy - 0.5 * h) * orig_h
            xmax = (cx + 0.5 * w) * orig_w
            ymax = (cy + 0.5 * h) * orig_h
            
            coco_predictions.append({
                "image_id": img_id,
                "category_id": int(label.item()),
                "bbox": [xmin.item(), ymin.item(), (xmax - xmin).item(), (ymax - ymin).item()],
                "score": float(score.item())
            })
            
        for t in thresholds:
            keep_t = all_scores > t
            t_scores, t_labels, t_boxes = all_scores[keep_t], all_labels[keep_t], pred_boxes[0, keep_t]
            
            frame_pred_boxes = []
            frame_pred_labels = []
            frame_pred_scores = []
            
            for score, label, box in zip(t_scores, t_labels, t_boxes):
                cx, cy, w, h = box.unbind(-1)
                xmin = (cx - 0.5 * w) * orig_w
                ymin = (cy - 0.5 * h) * orig_h
                xmax = (cx + 0.5 * w) * orig_w
                ymax = (cy + 0.5 * h) * orig_h
                
                frame_pred_boxes.append([xmin.item(), ymin.item(), xmax.item(), ymax.item()])
                frame_pred_labels.append(int(label.item()))
                frame_pred_scores.append(float(score.item()))
                
            hits, frame_gt, tp_confs, fp_confs = evaluate_containment(
                frame_pred_boxes, frame_pred_labels, frame_pred_scores, 
                gt_dict[img_id]["boxes"], gt_dict[img_id]["labels"]
            )
            
            containment_tracker[t]["hits"] += hits
            containment_tracker[t]["gt"] += frame_gt
            containment_tracker[t]["tp_confs"].extend(tp_confs)
            containment_tracker[t]["fp_confs"].extend(fp_confs)

    avg_latency = np.mean(inference_latencies) if len(inference_latencies) > 0 else 0
    avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    separator = "=" * 55
    log_print("\n" + separator, log_file)
    log_print("=== HARDWARE PERFORMANCE ===", log_file)
    log_print(separator, log_file)
    log_print(f"Average CPU Latency: {avg_latency:.2f} ms/frame", log_file)
    log_print(f"Estimated Throughput: {avg_fps:.2f} FPS", log_file)

    log_print("\n" + separator, log_file)
    log_print("=== CUSTOM CONTAINMENT METRIC (INSIDE-THE-BOX) ===", log_file)
    
    for t in thresholds:
        total_hits = containment_tracker[t]["hits"]
        total_gt = containment_tracker[t]["gt"]
        acc = (total_hits / total_gt) * 100 if total_gt > 0 else 0
        
        avg_tp_conf = np.mean(containment_tracker[t]["tp_confs"]) if containment_tracker[t]["tp_confs"] else 0.0
        avg_fp_conf = np.mean(containment_tracker[t]["fp_confs"]) if containment_tracker[t]["fp_confs"] else 0.0
        
        log_print(separator, log_file)
        log_print(f"--- Threshold {t:.2f} ---", log_file)
        log_print(f"Total Ground Truth: {total_gt}", log_file)
        log_print(f"Successful Center Hits: {total_hits}", log_file)
        log_print(f"Containment Hit Rate: {acc:.2f}%", log_file)
        log_print(f"Avg Conf (Accurate Guesses):   {avg_tp_conf:.4f}  (from {len(containment_tracker[t]['tp_confs'])} hits)", log_file)
        log_print(f"Avg Conf (False Positives):    {avg_fp_conf:.4f}  (from {len(containment_tracker[t]['fp_confs'])} errors)", log_file)
        
    log_print(separator, log_file)

    if not coco_predictions:
        log_print("No predictions generated for COCO mAP.", log_file)
    else:
        with open(temp_pred_json, "w") as f:
            json.dump(coco_predictions, f)
            
        log_print("\n=== COCO mAP EVALUATION ===", log_file)
        coco_dt = coco_gt.loadRes(temp_pred_json)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        
        original_stdout = sys.stdout
        sys.stdout = log_file
        coco_eval.summarize()
        sys.stdout = original_stdout
        coco_eval.summarize() 

    if os.path.exists(temp_gt_json):
        os.remove(temp_gt_json)
    if os.path.exists(temp_pred_json):
        os.remove(temp_pred_json)
    log_file.close()

if __name__ == "__main__":
    main()