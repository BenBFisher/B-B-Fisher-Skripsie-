import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F_vision

class RAMCachedCOCODataset(Dataset):
    def __init__(self, img_dir, label_dir, target_size=640, cache_in_ram=True):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.target_size = target_size
        self.cache_in_ram = cache_in_ram
        
        self.img_names = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.ram_cache = {}
        
        if self.cache_in_ram:
            print(f"Pre-allocating RAM cache for {len(self.img_names)} images...")

    def __len__(self):
        return len(self.img_names)
        
    def __getitem__(self, idx):
        # Return cached tensor if already loaded in RAM
        if idx in self.ram_cache:
            image_tensor, target = self.ram_cache[idx]
            return image_tensor, target

        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        # Load and preprocess image
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.target_size, self.target_size))
        
        image_tensor = F_vision.to_tensor(image)
        image_tensor = F_vision.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Parse labels
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        labels.append(int(parts[0]))
                        boxes.append([float(x) for x in parts[1:5]]) 
                        
        target = {
            "labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32)
        }
        
        # Store in RAM cache for future epochs
        if self.cache_in_ram:
            self.ram_cache[idx] = (image_tensor, target)
            
        return image_tensor, target

def custom_collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return pixel_values, targets