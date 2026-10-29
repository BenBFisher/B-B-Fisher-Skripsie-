import os
import torch
from PIL import Image
import torchvision.transforms.functional as F_vision

# Custom architecture imports from your original script
from local_config_rt_detr_v2 import RTDetrV2Config
from local_modeling_rt_detr_v2 import RTDetrV2ForObjectDetection 

def preprocess_image(image_path, target_size=32):
    """Loads, resizes to micro resolution, and normalizes using ImageNet stats."""
    image = Image.open(image_path).convert("RGB")
    orig_width, orig_height = image.size
    image_resized = image.resize((target_size, target_size))
    img_tensor = F_vision.to_tensor(image_resized)
    # ImageNet normalization matching your original benchmark
    img_tensor = F_vision.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return img_tensor.unsqueeze(0), orig_width, orig_height

def log_print(msg, file_handle):
    print(msg)
    file_handle.write(msg + "\n")
    file_handle.flush()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_file_path = "benchmark_torch_relaxed_results.txt"
    log_file = open(log_file_path, "w")
    
    log_print(f"--- Initializing PyTorch Relaxed Classification Benchmark on {device.upper()} ---", log_file)
    
    # Paths pulled from your original benchmark
    val_img_dir = "../phase2_2408_RT_DETR/coco_train2017/images/train"
    val_label_dir = "../phase2_2408_RT_DETR/coco_train2017_small1409_2/labels/train"
    weights_path = "micro_rtdetr_version_4full_epoch_50.pth"
    
    if not os.path.exists(val_img_dir) or not os.path.exists(val_label_dir):
        log_print("Error: Could not locate validation images or labels directory.", log_file)
        log_file.close()
        return

    # Match your custom micro architecture configuration exactly
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
    
    log_print(f"Loading weights from {weights_path} into custom architecture...", log_file)
    model = RTDetrV2ForObjectDetection(config).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    model.eval()

    valid_images = [f for f in sorted(os.listdir(val_img_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    total_valid_images = 0
    correct_top1_guesses = 0
    total_confidence = 0.0
    
    log_print(f"Evaluating images to see if the model knows WHAT it is looking at...", log_file)
    
    with torch.no_grad():
        for count, img_name in enumerate(valid_images):
            # 1. Parse ground truth directly from YOLO .txt files
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            txt_path = os.path.join(val_label_dir, txt_name)
            
            gt_classes = set()
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            class_id = int(parts[0])
                            if class_id < 2:  # Only evaluate targets 0 and 1
                                gt_classes.add(class_id)
                                
            # Skip images that don't contain our target classes
            if len(gt_classes) == 0:
                continue
                
            total_valid_images += 1
            
            # 2. Preprocess and run inference
            img_path = os.path.join(val_img_dir, img_name)
            pixel_values, _, _ = preprocess_image(img_path, target_size=32)
            pixel_values = pixel_values.to(device)
            
            # The model outputs a dictionary containing "logits" and "pred_boxes"
            outputs = model(pixel_values) 
            logits = outputs["logits"][0] # Extract raw class logits for the batch item
            
            # 3. Apply sigmoid and extract highest confidence guess
            scores_all = torch.sigmoid(logits)
            max_scores_per_query, class_labels_per_query = torch.max(scores_all, dim=1)
            top1_score, top1_index = torch.max(max_scores_per_query, dim=0)
            top1_class = class_labels_per_query[top1_index].item()
            
            total_confidence += top1_score.item()
            
            # 4. Check if the model's top guess exists in the image
            if top1_class in gt_classes:
                correct_top1_guesses += 1

            if total_valid_images > 0 and total_valid_images % 500 == 0:
                log_print(f"Processed {total_valid_images} images containing targets...", log_file)

    # Calculate metrics
    accuracy = (correct_top1_guesses / total_valid_images) * 100 if total_valid_images > 0 else 0
    avg_conf = (total_confidence / total_valid_images) * 100 if total_valid_images > 0 else 0
    
    summary = (
        f"\n{'='*50}\n"
        f"=== RELAXED CLASSIFICATION METRICS (CUSTOM PYTORCH) ===\n"
        f"{'='*50}\n"
        f"Total Images with Targets: {total_valid_images}\n"
        f"Correct Top-1 Class Guesses: {correct_top1_guesses}\n"
        f"Top-1 Classification Accuracy: {accuracy:.2f}%\n"
        f"Average Confidence of Top-1 Guess: {avg_conf:.2f}%\n"
        f"{'='*50}\n"
        f"Note: Evaluated strictly on label presence, ignoring bounding box overlap."
    )
    
    log_print(summary, log_file)
    log_file.close()

if __name__ == "__main__":
    main()