import cv2
import torch
import numpy as np
from safetensors.torch import load_file
from local_config_rt_detr_v2 import RTDetrV2Config
from local_modeling_rt_detr_v2 import RTDetrV2ForObjectDetection

# --- 1. CUSTOM PURE PYTORCH IMAGE PROCESSING ---

def preprocess_image(frame, target_size=640):
    """Resizes, normalizes, and converts a standard BGR OpenCV frame to a PyTorch tensor."""
    # Convert BGR to RGB and resize
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(rgb_frame, (target_size, target_size))
    
    # Convert to tensor and scale to [0, 1]
    img_tensor = torch.from_numpy(resized_img).permute(2, 0, 1).float() / 255.0
    
    # Normalize using standard ImageNet statistics
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    # Add batch dimension: [1, 3, 640, 640]
    return img_tensor.unsqueeze(0)

def post_process_boxes(logits, boxes, orig_width, orig_height, threshold=0.3):
    """Filters raw model outputs and scales bounding boxes back to screen size."""
    # Convert raw logits to probabilities
    probs = torch.sigmoid(logits[0]) 
    
    # Get highest scoring class for each of the 300 queries
    scores, labels = probs.max(dim=-1) 
    
    # Filter out anything below our confidence threshold
    keep = scores > threshold
    scores = scores[keep]
    labels = labels[keep]
    boxes = boxes[0, keep] 
    
    # RT-DETR outputs [Center_X, Center_Y, Width, Height] between 0 and 1.
    # Convert to [X_min, Y_min, X_max, Y_max]
    cx, cy, w, h = boxes.unbind(-1)
    xmin = cx - 0.5 * w
    ymin = cy - 0.5 * h
    xmax = cx + 0.5 * w
    ymax = cy + 0.5 * h
    
    # Scale back to the original webcam frame resolution
    xmin = xmin * orig_width
    xmax = xmax * orig_width
    ymin = ymin * orig_height
    ymax = ymax * orig_height
    
    out_boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
    
    return scores, labels, out_boxes

# --- 2. INITIALIZE ARCHITECTURE ---

print("1. Initializing pure PyTorch architecture...")
config = RTDetrV2Config()
model = RTDetrV2ForObjectDetection(config)

print("2. Loading and mapping weights...")
hf_state_dict = load_file("./my_local_model/model.safetensors")
new_state_dict = {}

for key, tensor in hf_state_dict.items():
    new_key = key
    if new_key.startswith("model."):
        new_key = new_key.replace("model.", "", 1)
    if new_key.startswith("backbone.") or new_key.startswith("denoising_class_embed"):
        continue
    if new_key.startswith("decoder.bbox_embed"):
        new_key = new_key.replace("decoder.bbox_embed", "bbox_embed")
    if new_key.startswith("decoder.class_embed"):
        new_key = new_key.replace("decoder.class_embed", "class_embed")
    new_state_dict[new_key] = tensor

model.load_state_dict(new_state_dict, strict=False)
model.eval()

# --- 3. LIVE WEBCAM INFERENCE ---

print("\n3. Starting live webcam feed... (Press 'q' to quit)")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
        
    orig_height, orig_width = frame.shape[:2]

    # Pre-process using our custom PyTorch function (No Hugging Face!)
    pixel_values = preprocess_image(frame)

    # Run inference
    with torch.no_grad():
        outputs_dict = model(pixel_values)
        
    # Post-process using our custom math
    scores, labels, boxes = post_process_boxes(
        outputs_dict["logits"], 
        outputs_dict["pred_boxes"], 
        orig_width, 
        orig_height, 
        threshold=0.3
    )

    # Draw the bounding boxes
    for score, label, box in zip(scores, labels, boxes):
        box = [int(i.item()) for i in box]
        class_name = config.id2label.get(label.item(), f"Class_{label.item()}")
        confidence = round(score.item(), 2)

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        label_text = f"{class_name}: {confidence}"
        cv2.putText(frame, label_text, (box[0], max(0, box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Pure PyTorch RT-DETR', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()