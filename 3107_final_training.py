import os
import math
import torch
import json
from PIL import Image
import torchvision.transforms.functional as F_vision
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
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
        
        valid_exts = ('.jpg', '.jpeg', '.png')
        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_name)
        
        image = Image.open(img_path).convert("RGB")
        image_resized = image.resize((self.target_size, self.target_size))
        
        img_tensor = F_vision.to_tensor(image_resized)
        img_tensor = F_vision.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        labels = []
        boxes = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        labels.append(class_id)
                        boxes.append([cx, cy, w, h])
                        
        if len(labels) == 0:
            labels = torch.zeros((0,), dtype=torch.long)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            labels = torch.tensor(labels, dtype=torch.long)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            
        # Reverted key back to "labels" for the custom local_loss.py script
        target = {"labels": labels, "boxes": boxes}
        return img_tensor, target


def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


# ==========================================
# 2. MAIN EXECUTION BLOCK
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Hero Run on: {device}")

    # 1. Load FULL Training Dataset
    dataset = CustomYOLODataset("./coco_train2017/images/train", "./coco_train2017/labels/train")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
    
# 2. Setup 80-Class Model & Load Pre-trained Weights (Matching Optuna)
    config = RTDetrV2Config(num_labels=80)
    model = RTDetrV2ForObjectDetection(config)
    
    print("Fetching pre-trained 80-class weights into RAM...")
    weights_path = hf_hub_download(repo_id="PekingU/rtdetr_v2_r18vd", filename="model.safetensors")
    hf_state_dict = load_file(weights_path)
    
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
    model.to(device)

    # 3. Freeze Backbone
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False

    # 4. Load the Winning Optuna Hyperparameters Automatically
    print("Loading ideal hyperparameters from Optuna...")
    try:
        with open("best_params.json", "r") as f:
            best_params = json.load(f)
    except FileNotFoundError:
        print("Error: best_params.json not found. Did the Optuna script run completely?")
        return

    EPOCHS = 20
    BEST_MAX_LR = 0.0004244450665673227
    BEST_PCT_START = 0.37977202786656683
    #BEST_MAX_LR = best_params["max_lr"]
    #BEST_PCT_START = best_params["pct_start"]


    

    print(f"Using Max LR: {BEST_MAX_LR:.6f} | Warmup: {BEST_PCT_START:.4f}")

    criterion = RTDetrLoss(num_classes=80)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=BEST_MAX_LR / 10, weight_decay=1e-4)
    
    accumulation_steps = 4
    
    
    steps_per_epoch = math.ceil(len(dataloader) / accumulation_steps)
    total_steps = steps_per_epoch * EPOCHS

    scheduler = OneCycleLR(
        optimizer,
        max_lr=BEST_MAX_LR,
        total_steps=total_steps,
        pct_start=BEST_PCT_START,
        anneal_strategy='cos'
    )

    
    # 5. The Training Loop
    print("\nStarting Training Loop...")
    
    # Open a text file in "write" mode
    log_file_path = "training_log_cocofull.txt"
    log_file = open(log_file_path, "w")
    log_file.write("--- RT-DETR COCO5k Training Log ---\n")
    
    model.train()
    for epoch in range(EPOCHS):
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
            
        # Format the output string
        avg_loss = epoch_loss / len(dataloader)
        log_message = f"Epoch {epoch + 1}/{EPOCHS} | Avg Loss: {avg_loss:.4f}"
        
        # Print to terminal AND write to the file
        print(log_message)
        log_file.write(log_message + "\n")
        log_file.flush() # Forces Python to save the file immediately so you can read it while it runs

    # Close the log file when finished
    log_file.close()
            
    

    # 6. Save the Final Model Weights
    save_path = "phase_1_weights_3107_cocofull.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining Complete! Weights successfully saved to {save_path}")

if __name__ == "__main__":
    main()