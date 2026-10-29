import torch
import os
from PIL import Image
import torchvision.transforms.functional as F_vision
from torch.utils.data import Dataset, DataLoader

# Model instantiation as demonstrated in your webcam_live.py
from modeling_rt_detr_v2 import RTDetrV2ForObjectDetection
# Custom loss function exactly as used in your 3107_final_training.py
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
            
        # Reverted key back to "labels" to match the custom RTDetrLoss expectations
        target = {"labels": labels, "boxes": boxes}
        return img_tensor, target

def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets

# ==========================================
# 2. VALIDATION EVALUATION SCRIPT
# ==========================================
def evaluate_validation_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating model on: {device.type.upper()}\n" + "="*45)

    # 1. Load Validation Dataset
    val_images_dir = "./coco5k/images/val"
    val_labels_dir = "./coco5k/labels/val"
    
    if not os.path.exists(val_images_dir):
        print(f"Directory not found: {val_images_dir}. Please update the path.")
        return

    val_dataset = CustomYOLODataset(val_images_dir, val_labels_dir)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)
    print(f"Found {len(val_dataset)} validation images.")

    # 2. Setup Model (using the implementation shown in your webcam_live.py)
    print("Loading local model from directory...")
    model = RTDetrV2ForObjectDetection.from_pretrained("./my_local_model")
    model.to(device)
    model.eval()

    # 3. Setup Custom Loss (Maintaining 80 classes)
    criterion = RTDetrLoss(num_classes=80)

    total_loss = 0.0
    total_batches = len(val_loader)

    print("\nRunning mathematical loss evaluation...")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            outputs = model(images)
            
            # Evaluate loss exactly like the final training script
            loss_dict = criterion(outputs, targets)
            batch_loss = loss_dict["loss_ce"] + loss_dict["loss_bbox"] + loss_dict["loss_giou"]
            
            total_loss += batch_loss.item()

    avg_loss = total_loss / total_batches
    print("\n" + "="*45)
    print(f"Final Validation Loss: {avg_loss:.4f}")
    print("="*45)

if __name__ == "__main__":
    evaluate_validation_loss()