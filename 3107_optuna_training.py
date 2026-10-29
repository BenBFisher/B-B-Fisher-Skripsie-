import os
import json
import torch
import math
import optuna
import subprocess
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from PIL import Image
import torchvision.transforms.functional as F_vision
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Import your custom local architecture files
from local_config_rt_detr_v2 import RTDetrV2Config
from local_modeling_rt_detr_v2 import RTDetrV2ForObjectDetection
from local_loss import RTDetrLoss

# ==========================================
# 1. DATASET & DATALOADER DEFINITION
# ==========================================
class CustomYOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, target_size=640):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.target_size = target_size
        
        # Grab all valid images in the folder
        valid_exts = ('.jpg', '.jpeg', '.png')
        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # YOLO pairs image123.jpg directly with image123.txt
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_name)
        
        # --- PROCESS IMAGE ---
        image = Image.open(img_path).convert("RGB")
        image_resized = image.resize((self.target_size, self.target_size))
        
        img_tensor = F_vision.to_tensor(image_resized)
        img_tensor = F_vision.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # --- PROCESS YOLO TXT FILE ---
        labels = []
        boxes = []
        
        # Read the normalized bounding boxes directly from the text file
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        labels.append(class_id)
                        boxes.append([cx, cy, w, h])
                        
        # Failsafe for empty images
        if len(labels) == 0:
            labels = torch.zeros((0,), dtype=torch.long)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            labels = torch.tensor(labels, dtype=torch.long)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            
        target = {"labels": labels, "boxes": boxes}

        # Add this line to verify it found the files:
        #print(f"DEBUG: Found {len(self.image_files)} images in {img_tensor.size()} and {len(labels)} labels in {label_path}")
        return img_tensor, target

        

def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets

# ==========================================
# 2. THE OPTUNA OBJECTIVE FUNCTION
# ==========================================
def objective(trial, dataloader, device, hf_state_dict):
    start_msg = f"\n--- Starting Trial {trial.number} ---"
    print(start_msg)
    
    # Safely append the start message to the log file
    with open("optuna_search_log.txt", "a") as f:
        f.write(start_msg + "\n")
    
    # Optuna actively searches for the perfect learning rate and warmup window
    suggested_max_lr = trial.suggest_float("max_lr", 1e-5, 5e-4, log=True)
    suggested_warmup = trial.suggest_float("pct_start", 0.1, 0.4)
    
    # Initialize a fresh 80-class model for this trial
    config = RTDetrV2Config(num_labels=80)
    model = RTDetrV2ForObjectDetection(config)
    
    # Format and load the pre-trained weights (Transfer Learning setup)
    new_state_dict = {}
    for key, tensor in hf_state_dict.items():
        new_key = key
        if new_key.startswith("model."):
            new_key = new_key.replace("model.", "", 1)
        if new_key.startswith("denoising_class_embed"):
            continue 
        if new_key.startswith("decoder.bbox_embed"):
            new_key = new_key.replace("decoder.bbox_embed", "bbox_embed")
        if new_key.startswith("decoder.class_embed"):
            new_key = new_key.replace("decoder.class_embed", "class_embed")
        new_state_dict[new_key] = tensor
        
    model.load_state_dict(new_state_dict, strict=False)
    
    # Freeze the ResNet-18 Backbone to save compute
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False
            
    model.to(device)

    # Setup Optimizer & Scheduler
    criterion = RTDetrLoss(num_classes=80)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=suggested_max_lr, weight_decay=1e-4)

    search_epochs = 20 
    accumulation_steps = 4
    steps_per_epoch = math.ceil(len(dataloader) / accumulation_steps)
    total_steps = steps_per_epoch * search_epochs
    
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=suggested_max_lr, 
        total_steps=total_steps, 
        pct_start=suggested_warmup, 
        anneal_strategy='cos'
    )

    # The Fast GPU Training Loop
    model.train()
    for epoch in range(search_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            loss_dict = criterion(outputs, targets)

            # Dynamically sum all primary and auxiliary loss components
            total_loss = sum(value for key, value in loss_dict.items() if 'loss' in key)

            loss = total_loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(dataloader)
        
        # Format, print, and append the epoch results
        log_message = f"Trial {trial.number} | Epoch {epoch+1}/{search_epochs} | Avg Loss: {avg_loss:.4f}"
        print(log_message)
        with open("optuna_search_log.txt", "a") as f:
            f.write(log_message + "\n")
        
        # Kill the trial early if the loss starts climbing (mode collapse)
        trial.report(avg_loss, epoch)
        if trial.should_prune():
            prune_msg = f"Loss diverged. Optuna is pruning Trial {trial.number} early."
            print(prune_msg)
            with open("optuna_search_log.txt", "a") as f:
                f.write(prune_msg + "\n")
            raise optuna.TrialPruned()

    return avg_loss

# ==========================================
# 3. MAIN EXECUTION BLOCK
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device.type.upper()}")

    print("Fetching pre-trained 80-class weights into RAM...")
    weights_path = hf_hub_download(repo_id="PekingU/rtdetr_v2_r18vd", filename="model.safetensors")
    hf_state_dict = load_file(weights_path)

    print("Loading dataset...")
    # Updated to the correct training paths
    dataset = CustomYOLODataset(
        images_dir="./coco5k/images/val", 
        labels_dir="./coco5k/labels/val"
    )
    
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, 
        collate_fn=custom_collate_fn, num_workers=2, pin_memory=True
    )

    print("\nStarting Optuna Hyperparameter Search...")
    
    # Initialize the log file with a clean header
    with open("optuna_search_log.txt", "w") as f:
        f.write("=== Optuna Hyperparameter Search Log ===\n")

    func = lambda trial: objective(trial, dataloader, device, hf_state_dict)
    
    study = optuna.create_study(
        study_name="rt_detr_coco5k_run",
        storage="sqlite:///optuna_coco5k.db", 
        load_if_exists=True,
        direction="minimize"
    )

    study.optimize(func, n_trials=20)

    # Log the final results
    summary_msg = (
        f"\n=== OPTIMIZATION COMPLETE ===\n"
        f"Best Trial: {study.best_trial.number}\n"
        f"Lowest Loss: {study.best_trial.value}\n"
        f"Best Hyperparameters:\n"
    )
    for key, value in study.best_trial.params.items():
        summary_msg += f"    {key}: {value}\n"
        
    print(summary_msg)
    with open("optuna_search_log.txt", "a") as f:
        f.write(summary_msg)
    

    # 1. Log the final results
    summary_msg = (
        f"\n=== OPTIMIZATION COMPLETE ===\n"
        f"Best Trial: {study.best_trial.number}\n"
        f"Lowest Loss: {study.best_trial.value}\n"
        f"Best Hyperparameters:\n"
    )
    for key, value in study.best_trial.params.items():
        summary_msg += f"    {key}: {value}\n"
        
    print(summary_msg)
    with open("optuna_search_log.txt", "a") as f:
        f.write(summary_msg)

    # 2. Save the winning numbers to a file for the next script
    print("\nSaving winning hyperparameters to 'best_params.json'...")
    with open("best_params.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)

    # 3. Automatically trigger the final training script
    print("Launching final Hero Run...\n" + "="*45)
    subprocess.run(["py", "3107_final_training.py"])

if __name__ == "__main__":
    main()