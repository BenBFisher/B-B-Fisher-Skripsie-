import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F_vision
import onnxruntime as ort

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
        
        # Parse ground truth classes for the relaxed metric
        gt_classes = set()
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # Only keep target classes 0 (person) and 1 (chair)
                        if class_id < 2:
                            gt_classes.add(class_id)
                            
        return img_numpy, gt_classes

# ==========================================
# 2. MAIN EVALUATION BLOCK
# ==========================================
def main():
    onnx_model_path = "micro_rtdetr_32_static_4.onnx"
    log_file_path = "benchmark_onnx_relaxed_results_32.txt"
    log_file = open(log_file_path, "w")
    
    log_print(f"--- Initializing ONNX Relaxed Accuracy Benchmark ---", log_file)
    log_print(f"Loading {onnx_model_path} with CPUExecutionProvider...", log_file)
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_model_path, sess_options, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Paths pointing to the parent directory
    val_images_path = "../phase2_2408_RT_DETR/coco_train2017/images/train"
    val_labels_path = "../phase2_2408_RT_DETR/coco_train2017_small1409_2/labels/train"
    
    dataset = ONNXYOLODataset(val_images_path, val_labels_path, target_size=32)
    
    total_valid_images = 0
    correct_top1_guesses = 0
    total_confidence = 0.0
    
    log_print(f"Evaluating {len(dataset)} images via ONNX Runtime for Class Accuracy...", log_file)
    
    for i in range(len(dataset)):
        img_np, gt_classes = dataset[i]
        
        # Skip images that don't contain our targets
        if len(gt_classes) == 0:
            continue
            
        total_valid_images += 1
        
        outputs = session.run(None, {input_name: img_np})
        
        # Extract the raw logits from the first output tensor
        logits = torch.tensor(outputs[0][0])
        
        # Convert to probability and find the most confident guess
        scores_all = torch.sigmoid(logits)
        max_scores_per_query, class_labels_per_query = torch.max(scores_all, dim=1)
        top1_score, top1_index = torch.max(max_scores_per_query, dim=0)
        top1_class = class_labels_per_query[top1_index].item()
        
        total_confidence += top1_score.item()
        
        if top1_class in gt_classes:
            correct_top1_guesses += 1
            
        if total_valid_images > 0 and total_valid_images % 500 == 0:
            log_print(f"Processed {total_valid_images} valid target frames...", log_file)

    # Compute final metrics
    accuracy = (correct_top1_guesses / total_valid_images) * 100 if total_valid_images > 0 else 0
    avg_conf = (total_confidence / total_valid_images) * 100 if total_valid_images > 0 else 0
    
    separator = "=" * 45
    log_print("\n" + separator, log_file)
    log_print("=== ONNX RELAXED CLASSIFICATION METRICS ===", log_file)
    log_print(separator, log_file)
    log_print(f"Total Images with Targets: {total_valid_images}", log_file)
    log_print(f"Correct Top-1 Guesses:     {correct_top1_guesses}", log_file)
    log_print(f"Classification Accuracy:   {accuracy:.2f}%", log_file)
    log_print(f"Average Top-1 Confidence:  {avg_conf:.2f}%", log_file)
    log_print(separator, log_file)
    
    log_file.close()
    print(f"Results successfully saved to {log_file_path}")

if __name__ == "__main__":
    main()