import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as F_vision

class COCO128Dataset(Dataset):
    def __init__(self, img_dir, label_dir, target_size=640):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.target_size = target_size
        
        # Grab all image filenames and sort them to ensure consistent ordering
        self.img_names = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        
    def __len__(self):
        return len(self.img_names)
        
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # The label file has the exact same name, just with a .txt extension
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        # 1. Load and preprocess the image
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.target_size, self.target_size)) # Force to 640x640
        
        # Convert to tensor and apply standard ImageNet normalization
        image_tensor = F_vision.to_tensor(image)
        image_tensor = F_vision.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # 2. Parse the labels (Format: class_id, center_x, center_y, width, height)
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        labels.append(int(parts[0]))
                        # YOLO/COCO128 format is already normalized between 0 and 1!
                        boxes.append([float(x) for x in parts[1:5]]) 
                        
        # 3. Package the target dictionaries
        target = {
            "labels": torch.tensor(labels, dtype=torch.long),
            # If an image has no objects, return an empty 0x4 tensor to prevent crashes
            "boxes": torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32)
        }
        
        return image_tensor, target

def custom_collate_fn(batch):
    """
    Safely batches images and targets. 
    Images are stacked into a single tensor, but targets are kept as a list of dictionaries 
    since they have different lengths.
    """
    pixel_values = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return pixel_values, targets

# --- Quick Validation Test ---
if __name__ == "__main__":
    # Define paths based on the COCO128 extraction
    IMG_DIR = "./coco128/images/train2017"
    LABEL_DIR = "./coco128/labels/train2017"
    
    print("1. Initializing Dataset...")
    dataset = COCO128Dataset(img_dir=IMG_DIR, label_dir=LABEL_DIR)
    print(f"Loaded {len(dataset)} images.")
    
    print("\n2. Initializing DataLoader (Batch Size: 2)...")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)
    
    print("\n3. Testing the first batch...")
    for images, targets in dataloader:
        print(f"Batch Image Tensor Shape: {images.shape}")
        
        for i, target in enumerate(targets):
            print(f"  - Image {i+1} in batch contains {len(target['labels'])} objects.")
            print(f"    Box Tensor Shape: {target['boxes'].shape}")
        
        print("\nSUCCESS: Dataset and DataLoader are fully operational.")
        break # Just test the first batch and exit