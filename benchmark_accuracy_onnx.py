import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F_vision
import onnxruntime as ort
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def log_print(msg, file_handle):
    print(msg)
    file_handle.write(msg + "\n")
    file_handle.flush()

# ==========================================
# 1. DATASET DEFINITION 
# ==========================================
class ONNXYOLODataset:
    def __init__(self, images_dir, labels_dir, target_size=32):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.target_size = target_size
        valid_exts = ('.jpg', '.jpeg', '.png')
        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, os.path.splitext(img_name)[0] + ".txt")
        
        # Image Processing (must match PyTorch exactly)
        image = Image.open(img_path).convert("RGB")
        image_resized = image.resize((self.target_size, self.target_size))
        img_tensor = F_vision.to_tensor(image_resized)
        img_tensor = F_vision.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Convert to numpy array and add batch dimension for ONNX: (1, 3, 32, 32)
        img_numpy = img_tensor.unsqueeze(0).numpy()
        
        # Label Processing
        labels, boxes = [], []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # Only keep target classes 0 (person) and 1 (chair)
                        if class_id < 2:
                            labels.append(class_id)
                            boxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                        
        targets = {
            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        }
        return img_numpy, targets

# ==========================================
# 2. MAIN EVALUATION BLOCK
# ==========================================
def main():
    onnx_model_path = "micro_rtdetr_32_static_4.onnx"
    log_file_path = "benchmark_onnx_results_32.txt"
    log_file = open(log_file_path, "w")
    
    log_print(f"--- Initializing ONNX Accuracy Benchmark ---", log_file)
    log_print(f"Loading {onnx_model_path} with CPUExecutionProvider...", log_file)
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_model_path, sess_options, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Paths pointing to the parent directory
    val_images_path = "../phase2_2408_RT_DETR/coco_train2017/images/train"
    val_labels_path = "../phase2_2408_RT_DETR/coco_train2017_small1409_2/labels/train"
    
    dataset = ONNXYOLODataset(val_images_path, val_labels_path, target_size=32)
    metric = MeanAveragePrecision(box_format='cxcywh', iou_type='bbox')
    
    log_print(f"Evaluating {len(dataset)} images via ONNX Runtime...", log_file)
    
    for i in range(len(dataset)):
        if (i + 1) % 500 == 0:
            log_print(f"Processed {i + 1}/{len(dataset)} frames...", log_file)
            
        img_np, target = dataset[i]
        
        outputs = session.run(None, {input_name: img_np})
        
        logits = torch.tensor(outputs[0][0])
        pred_boxes = torch.tensor(outputs[1][0])
        
        scores_all = torch.sigmoid(logits)
        scores, labels = torch.max(scores_all, dim=1)
        
        # Threshold of 0.001 to build the full PR curve
        keep_idx = scores > 0.001 
        
        preds = [{
            "boxes": pred_boxes[keep_idx],
            "scores": scores[keep_idx],
            "labels": labels[keep_idx]
        }]
        
        targets = [target]
        metric.update(preds, targets)

    log_print("\nComputing Mean Average Precision (mAP)...", log_file)
    results = metric.compute()
    
    separator = "=" * 45
    log_print("\n" + separator, log_file)
    log_print("=== ONNX ACCURACY METRICS (COCO mAP) ===", log_file)
    log_print(separator, log_file)
    log_print(f"mAP (IoU=0.50:0.95): {results['map'].item():.4f}", log_file)
    log_print(f"mAP (IoU=0.50):      {results['map_50'].item():.4f}", log_file)
    log_print(f"mAP (IoU=0.75):      {results['map_75'].item():.4f}", log_file)
    log_print(separator, log_file)
    
    log_file.close()
    print(f"Results successfully saved to {log_file_path}")

if __name__ == "__main__":
    main()