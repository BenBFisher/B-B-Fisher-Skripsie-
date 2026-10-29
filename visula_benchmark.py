import torch
import cv2
import os
from PIL import Image
import torchvision.transforms.functional as F_vision

# Import your configuration and the model architecture
from transformers import RTDetrImageProcessor
from configuration_rt_detr_v2 import RTDetrV2Config
from modeling_rt_detr_v2 import RTDetrV2ForObjectDetection
#from local_config_rt_detr_v2 import RTDetrV2Config
#from local_modeling_rt_detr_v2 import RTDetrV2ForObjectDetection

def preprocess_image(image_path, target_size=640):
    """Loads, resizes, and normalizes a real image for the model."""
    image = Image.open(image_path).convert("RGB")
    orig_width, orig_height = image.size
    
    image_resized = image.resize((target_size, target_size))
    img_tensor = F_vision.to_tensor(image_resized)
    img_tensor = F_vision.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return img_tensor.unsqueeze(0), orig_width, orig_height

def post_process_boxes(logits, boxes, orig_width, orig_height, threshold=0.3):
    """Filters low-confidence boxes and scales coordinates back to the original image."""
    probs = torch.sigmoid(logits[0]) 
    scores, labels = probs.max(dim=-1) 
    
    keep = scores > threshold
    scores = scores[keep]
    labels = labels[keep]
    boxes = boxes[0, keep] 
    
    # Convert cx, cy, w, h to xmin, ymin, xmax, ymax
    cx, cy, w, h = boxes.unbind(-1)
    xmin = (cx - 0.5 * w) * orig_width
    ymin = (cy - 0.5 * h) * orig_height
    xmax = (cx + 0.5 * w) * orig_width
    ymax = (cy + 0.5 * h) * orig_height
    
    out_boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
    return scores, labels, out_boxes

def generate_visual_outputs(input_dir="./coco128/images/train2017", threshold=0.001):
    print("--- Initializing Visual Output Generator ---")
    
    # 1. Setup Output Directory
    output_dir = "benchmark_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading local model for live stream...")
    model = RTDetrV2ForObjectDetection.from_pretrained("./my_local_model")
    processor = RTDetrImageProcessor.from_pretrained("./my_local_model")
    model.eval()
    
    # 2. Initialize Architecture and Load Weights
    #print("Initializing model architecture...")
    #config = RTDetrV2Config(num_labels=80) 
    #model = RTDetrV2ForObjectDetection(config)
    
    #print("Loading custom trained weights...")
    #Safe CPU loading for local testing
    #model.load_state_dict(torch.load("rt_detr_custom_trained.pth", map_location=torch.device('cpu')))
    model.eval()

    # 3. Retrieve Test Images
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} not found.")
        return
        
    test_images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not test_images:
        print(f"Error: No images found in {input_dir}.")
        return

    print(f"Processing {len(test_images)} images. Saving results to ./{output_dir}/")

    # 4. Processing Loop
    with torch.no_grad():
        for img_name in test_images:
            img_path = os.path.join(input_dir, img_name)
            
            # Prepare image
            pixel_values, orig_width, orig_height = preprocess_image(img_path)
            
            # Forward pass
            outputs_dict = model(pixel_values)
            
            # Extract final reference points and logits
            scores, labels, boxes = post_process_boxes(
                outputs_dict["logits"],
                outputs_dict["pred_boxes"],
                orig_width, 
                orig_height, 
                threshold=threshold
            )
            
            # 5. Draw Bounding Boxes
            cv2_image = cv2.imread(img_path)
            
            for score, label, box in zip(scores, labels, boxes):
                box = [int(i.item()) for i in box]
                class_name = config.id2label.get(label.item(), f"Class_{label.item()}")
                confidence = round(score.item(), 2)
                
                # Draw Box
                cv2.rectangle(cv2_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
                # Draw Label Background (for readability)
                label_text = f"{class_name}: {confidence}"
                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(cv2_image, (box[0], box[1] - text_height - 10), (box[0] + text_width, box[1]), (0, 255, 0), -1)
                
                # Draw Label Text
                cv2.putText(cv2_image, label_text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # 6. Save the Processed Image
            save_path = os.path.join(output_dir, f"out_{img_name}")
            cv2.imwrite(save_path, cv2_image)
            print(f"Saved: {save_path} ({len(scores)} detections)")

    print(f"\nVisual generation complete. All images saved to ./{output_dir}/")

if __name__ == "__main__":
    generate_visual_outputs()