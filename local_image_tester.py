import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import RTDetrImageProcessor
from local_config_rt_detr_v2 import RTDetrV2Config
from local_modeling_rt_detr_v2 import RTDetrV2ForObjectDetection

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

print("3. Loading image processor...")
# Keep the HF processor to handle the annoying image math!
processor = RTDetrImageProcessor.from_pretrained("./my_local_model")

# --- INFERENCE PIPELINE ---

image_path = "idk.png"  
try:
    image = Image.open(image_path).convert("RGB")
    print(f"\nSuccessfully loaded '{image_path}'!")
except FileNotFoundError:
    print(f"Error: Could not find '{image_path}'. Make sure the image is in the same folder.")
    exit()

# Preprocess the image for the model
inputs = processor(images=image, return_tensors="pt")

# Run inference
with torch.no_grad():
    # Pass ONLY the pixel_values tensor to our custom model
    outputs_dict = model(inputs["pixel_values"])

# Create a tiny dummy class to bridge our dictionary to the HF post-processor
class MockOutput:
    def __init__(self, logits, pred_boxes):
        self.logits = logits
        self.pred_boxes = pred_boxes

outputs = MockOutput(outputs_dict["logits"], outputs_dict["pred_boxes"])

# Post-process the results (filtering detections with a 0.5 confidence threshold)
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, threshold=0.5, target_sizes=target_sizes
)[0]

# Print out what the model detected
print(f"\n--- Detection Results ---")
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    class_name = config.id2label.get(label.item(), f"Class_{label.item()}")
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected **{class_name}** with confidence **{round(score.item(), 3)}** at bounding box: {box}")