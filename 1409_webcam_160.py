import cv2
import time
import torch
from PIL import Image
import torchvision.transforms.functional as F_vision

# Import your custom local architecture files
from local_config_rt_detr_v2 import RTDetrV2Config
from local_modeling_rt_detr_v2 import RTDetrV2ForObjectDetection

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_WEIGHTS = "micro_rtdetr_version_3_epoch_200.pth"
CONFIDENCE_THRESHOLD = 0.31  # Adjust this if it is too strict or too noisy
INPUT_SIZE = 160

# Optional: Map your 10 class IDs to actual names for the display
# Replace these with your actual wildlife/target class names
CLASS_NAMES = {
    0: 'chair',
    1: 'person'
}

def main():
    # 1. Initialize Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Micro RT-DETR on: {device.type.upper()}")

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
    
    model = RTDetrV2ForObjectDetection(config)
    
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        print("Weights loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Could not find {MODEL_WEIGHTS}. Make sure your training script finished.")
        return

    model.to(device)
    model.eval() # CRITICAL: Switch from training to evaluation mode

    # 2. Initialize Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\nStarting webcam feed. Press 'q' to quit.")
    
    prev_time = time.time()

    # 3. Main Inference Loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get actual webcam resolution to scale boxes properly later
        frame_h, frame_w, _ = frame.shape

        # --- PREPROCESSING ---
        # OpenCV captures in BGR, PyTorch expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Resize to your model's strict 160x160 requirement
        resized_image = pil_image.resize((INPUT_SIZE, INPUT_SIZE))
        img_tensor = F_vision.to_tensor(resized_image)
        img_tensor = F_vision.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Add batch dimension and send to GPU
        input_batch = img_tensor.unsqueeze(0).to(device)

        # --- INFERENCE ---
        with torch.no_grad():
            outputs = model(input_batch)
            
        # Extract the final decoder layer predictions
        logits = outputs["logits"][0] # Shape: [num_queries, num_classes]
        boxes = outputs["pred_boxes"][0] # Shape: [num_queries, 4] normalized [cx, cy, w, h]
        
        # Apply sigmoid to convert raw logits into probabilities (0.0 to 1.0)
        probs = torch.sigmoid(logits)
        max_probs, labels = torch.max(probs, dim=-1)

        # --- POST-PROCESSING & RENDERING ---
        for prob, label, box in zip(max_probs, labels, boxes):
            if prob.item() > CONFIDENCE_THRESHOLD and label.item() != 4:  # Skip background class (ID 1)
                # Convert normalized [cx, cy, w, h] to absolute [xmin, ymin, xmax, ymax]
                cx, cy, w, h = box.cpu().numpy()
                
                xmin = int((cx - w / 2) * frame_w)
                ymin = int((cy - h / 2) * frame_h)
                xmax = int((cx + w / 2) * frame_w)
                ymax = int((cy + h / 2) * frame_h)
                
                # Draw the bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                
                # Draw the label and confidence score
                class_name = CLASS_NAMES.get(label.item(), f"ID {label.item()}")
                label_text = f"{class_name}: {prob.item():.2f}"
                cv2.putText(frame, label_text, (xmin, max(ymin - 10, 0)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- FPS CALCULATION ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the feed
        cv2.imshow('Micro RT-DETR Webcam', frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()