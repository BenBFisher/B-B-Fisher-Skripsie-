import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_convert
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import RTDetrV2ForObjectDetection, AutoImageProcessor

# ==========================================
# 1. THE YOLO 80-CLASS REFERENCE
# ==========================================
YOLO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

class CustomYOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
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
        orig_size = torch.tensor([image.height, image.width])
        
        labels = []
        boxes = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        labels.append(int(parts[0]))
                        boxes.append([float(x) for x in parts[1:]])
                        
        if len(labels) == 0:
            labels = torch.zeros((0,), dtype=torch.long)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            labels = torch.tensor(labels, dtype=torch.long)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            
        target = {"labels": labels, "boxes": boxes, "orig_size": orig_size}
        return image, target

def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

# ==========================================
# 2. EVALUATION METRIC EXECUTION
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Calculating True HF Baseline mAP on: {device.type.upper()}")

    dataset = CustomYOLODataset("./coco5k/images/val", "./coco5k/labels/val")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

    print("Downloading/Loading pure pre-trained HF weights and processor...")
    processor = AutoImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd", do_normalize=False, do_rescale=True)
    model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd").to(device)
    model.eval()

    # Create the dynamic Translation Map from HF strings to YOLO integers
    hf_to_yolo_map = {}
    for hf_id, name in model.config.id2label.items():
        clean_name = name.lower().replace("_", " ").replace(" ", "")
        
        # Find the matching YOLO class ignoring spaces (e.g., "tvmonitor" vs "tv")
        yolo_match = -1
        for y_idx, y_name in enumerate(YOLO_CLASSES):
            y_clean = y_name.replace(" ", "")
            if clean_name == y_clean or (clean_name == "tvmonitor" and y_clean == "tv"):
                yolo_match = y_idx
                break
        
        hf_to_yolo_map[hf_id] = yolo_match

    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=False)

    print("\nRunning inferences with active Class Translation... This may take a few minutes.")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            inputs = processor(images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            
            target_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.05)

            formatted_preds = []
            formatted_targets = []

            for i in range(len(images)):
                raw_hf_labels = results[i]["labels"]
                
                # Translate the labels!
                translated_labels = torch.tensor(
                    [hf_to_yolo_map[l.item()] for l in raw_hf_labels], 
                    device=device, dtype=torch.long
                )
                
                # Filter out anything the HF model predicted that isn't in the YOLO 80
                valid_mask = translated_labels != -1
                
                formatted_preds.append({
                    "boxes": results[i]["boxes"][valid_mask],
                    "scores": results[i]["scores"][valid_mask],
                    "labels": translated_labels[valid_mask],
                })

                tgt_labels = targets[i]["labels"].to(device)
                tgt_boxes = targets[i]["boxes"].to(device)
                
                if len(tgt_boxes) > 0:
                    tgt_boxes = box_convert(tgt_boxes, in_fmt="cxcywh", out_fmt="xyxy")
                    h, w = targets[i]["orig_size"].tolist()
                    tgt_boxes[:, [0, 2]] *= w
                    tgt_boxes[:, [1, 3]] *= h
                else:
                    tgt_boxes = torch.empty((0, 4), device=device)

                formatted_targets.append({
                    "boxes": tgt_boxes,
                    "labels": tgt_labels,
                })

            metric.update(formatted_preds, formatted_targets)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataloader)} batches...")

    print("\nFinalizing mAP calculations...")
    results = metric.compute()
    
    print("\n" + "="*45)
    print("=== TRUE BASELINE HF ACCURACY METRICS ===")
    print("="*45)
    print(f"mAP (IoU=0.50:0.95): {results['map'].item():.4f}")
    print(f"mAP (IoU=0.50):      {results['map_50'].item():.4f}")
    print(f"mAP (IoU=0.75):      {results['map_75'].item():.4f}")
    print(f"mAP (Small objects): {results['map_small'].item():.4f}")
    print(f"mAP (Large objects): {results['map_large'].item():.4f}")
    print("="*45)

if __name__ == "__main__":
    main()