import cv2
import torch
import os
from PIL import Image
import torchvision.transforms.functional as F_vision

# Import your local, standalone architecture
from local_config_rt_detr_v2 import RTDetrV2Config
from local_modeling_rt_detr_v2 import RTDetrV2ForObjectDetection

# Your exact custom coco2017 class mapping
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light", 
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 
    15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow", 
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack", 
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee", 
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat", 
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle", 
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 
    45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 
    55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven", 
    70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock", 
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
}

def preprocess_image(frame, target_size=640):
    """OpenCV captures in BGR; convert to RGB, resize, and normalize for PyTorch."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    orig_width, orig_height = image.size
    
    image_resized = image.resize((target_size, target_size))
    img_tensor = F_vision.to_tensor(image_resized)
    img_tensor = F_vision.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return img_tensor.unsqueeze(0), orig_width, orig_height

def post_process_boxes(logits, boxes, orig_width, orig_height, threshold=0.5):
    """Applies sigmoid activation, filters by confidence, and scales boxes to the webcam resolution."""
    probs = torch.sigmoid(logits[0]) 
    scores, labels = probs.max(dim=-1) 
    
    keep = scores > threshold
    scores = scores[keep]
    labels = labels[keep]
    boxes = boxes[0, keep] 
    
    # Convert normalized cx, cy, w, h back to absolute xmin, ymin, xmax, ymax
    cx, cy, w, h = boxes.unbind(-1)
    xmin = (cx - 0.5 * w) * orig_width
    ymin = (cy - 0.5 * h) * orig_height
    xmax = (cx + 0.5 * w) * orig_width
    ymax = (cy + 0.5 * h) * orig_height
    
    out_boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
    return scores, labels, out_boxes

def run_local_live_feed():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Initializing Local Architecture on {device.type.upper()} ---")

    # 1. Initialize strictly 80-class local model
    config = RTDetrV2Config(num_labels=80)
    model = RTDetrV2ForObjectDetection(config)

    # 2. Robust Weights Loader
    weights_path = "phase_1_weights_3107_cocofull.pth"
    print(f"Loading custom weights from: {weights_path}")
    
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        cleaned_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("model.", "", 1) if k.startswith("model.") else k
            cleaned_dict[new_key] = v
            
        model.load_state_dict(cleaned_dict, strict=False)
    else:
        print(f"Error: Could not find weights at {weights_path}")
        return

    model.to(device)
    model.eval()

    # 3. Start Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n=== Unrestricted Local Feed Running (Press 'q' to quit) ===")

    # 4. Processing Loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Preprocess and send to GPU
        pixel_values, orig_width, orig_height = preprocess_image(frame)
        pixel_values = pixel_values.to(device)

        # Forward Pass through local architecture
        with torch.no_grad():
            outputs_dict = model(pixel_values)
            
        # Post-process the raw output dictionary
        scores, labels, boxes = post_process_boxes(
            outputs_dict["logits"], 
            outputs_dict["pred_boxes"], 
            orig_width, 
            orig_height, 
            threshold=0.5
        )

        # 5. Draw Everything Detected
        for score, label, box in zip(scores, labels, boxes):
            class_id = label.item()
            
            # Fetch the name from our COCO_CLASSES dictionary
            class_name = COCO_CLASSES.get(class_id, f"Class_{class_id}")
            
            box = [int(i.item()) for i in box]
            confidence = round(score.item(), 2)

            # Draw Rectangle
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # Draw Label Background and Text
            label_text = f"{class_name}: {confidence}"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (box[0], box[1] - text_h - 10), (box[0] + text_w, box[1]), (0, 255, 0), -1)
            cv2.putText(frame, label_text, (box[0], max(0, box[1] - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow("RT-DETR Local Live Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_local_live_feed()