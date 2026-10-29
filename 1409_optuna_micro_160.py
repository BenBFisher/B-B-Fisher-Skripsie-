import os
import math
import torch
import json
import optuna
from PIL import Image
import torchvision.transforms.functional as F_vision
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from local_config_rt_detr_v2 import RTDetrV2Config
from local_modeling_rt_detr_v2 import RTDetrV2ForObjectDetection
from local_loss import RTDetrLoss

# ==========================================
# 1. DATASET DEFINITION
# ==========================================
class CustomYOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, target_size=160):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.target_size = target_size
        valid_exts = ('.jpg', '.jpeg', '.png')
        
        #print(f"Scanning {images_dir} for valid images and labels...", flush=True)  
        raw_files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)]
        self.image_files = []
        
        for f in raw_files:
            label_name = os.path.splitext(f)[0] + ".txt"
            label_path = os.path.join(self.labels_dir, label_name)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as file:
                    for line in file:
                        if len(line.strip().split()) == 5:
                            self.image_files.append(f)
                            break
                            
        #print(f"Done! Found {len(self.image_files)} valid images with targets.", flush=True)

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
        
        labels, boxes = [], []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append(int(parts[0]))
                    boxes.append(list(map(float, parts[1:])))
                        
        labels = torch.tensor(labels, dtype=torch.long)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        return img_tensor, {"labels": labels, "boxes": boxes}

def custom_collate_fn(batch):
    return torch.stack([item[0] for item in batch], dim=0), [item[1] for item in batch]


# ==========================================
# 2. LOAD DATASET ONCE GLOBALLY
# ==========================================
# Instantiating here ensures the disk read happens once, not on every trial
GLOBAL_DATASET = CustomYOLODataset("./coco5k/images/val", "./coco5k_small1409/labels/val", target_size=160)


# ==========================================
# 3. OPTUNA OBJECTIVE FUNCTION
# ==========================================
def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Starting Trial {trial.number} on {device} ---", flush=True)
    
    max_lr = trial.suggest_float("max_lr", 1e-4, 5e-3, log=True)
    pct_start = trial.suggest_float("pct_start", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    print(f"Trial {trial.number} Params -> max_lr: {max_lr:.6f}, pct_start: {pct_start:.3f}, weight_decay: {weight_decay:.6f}", flush=True)
    
    # num_workers=0 avoids multi-processing deadlocks on Windows
    dataloader = DataLoader(GLOBAL_DATASET, batch_size=32, shuffle=True, collate_fn=custom_collate_fn, num_workers=2)
    
    config = RTDetrV2Config(
        num_labels=10,
        hidden_dim=32,
        d_model=32,
        encoder_hidden_dim=32,
        decoder_layers=2,
        encoder_in_channels=(24, 48, 96), 
        decoder_in_channels=(32, 32, 32),
        encoder_ffn_dim=128,
        decoder_ffn_dim=128
    )
    
    model = RTDetrV2ForObjectDetection(config).to(device)
    
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False
            
    criterion = RTDetrLoss(num_classes=10)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=max_lr / 10, weight_decay=weight_decay)
    
    EPOCHS = 20
    accumulation_steps = 1
    steps_per_epoch = math.ceil(len(dataloader) / accumulation_steps)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=steps_per_epoch * EPOCHS,
        pct_start=pct_start,
        anneal_strategy='cos'
    )
    
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            total_loss = sum(value for key, value in loss_dict.items() if 'loss' in key)
            loss = total_loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            epoch_loss += total_loss.item()
            
            # Print feedback on the very first batch so you know it is running
            if batch_idx == 0 and epoch == 0:
                print(f"Trial {trial.number}: First batch processed successfully.", flush=True)
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Trial {trial.number} | Epoch {epoch + 1}/{EPOCHS} -> Loss: {avg_loss:.4f}", flush=True)
        
        trial.report(avg_loss, epoch)
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch + 1}.", flush=True)
            raise optuna.exceptions.TrialPruned()
            
    return avg_loss

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("\nBest Trial:", flush=True)
    trial = study.best_trial
    print(f"  Final Loss: {trial.value}", flush=True)
    for key, value in trial.params.items():
        print(f"    {key}: {value}", flush=True)

    with open("best_params_micro.json", "w") as f:
        json.dump(trial.params, f, indent=4)
    print("\nSaved optimal hyperparameters to best_params_micro.json", flush=True)