import torch
from PIL import Image
from configuration_rt_detr_v2 import RTDetrV2Config
from modeling_rt_detr_v2 import RTDetrV2ForObjectDetection
from transformers import RTDetrImageProcessor

print("Loading local architecture and weights...")
model = RTDetrV2ForObjectDetection.from_pretrained("./my_local_model")
processor = RTDetrImageProcessor.from_pretrained("./my_local_model")
model.eval()

# 1. Drop in your image file name here (e.g., 'test.jpg' or an animal photo)
image_path = "idk.png"  

try:
    image = Image.open(image_path).convert("RGB")
    print(f"Successfully loaded '{image_path}'!")
except FileNotFoundError:
    print(f"Error: Could not find '{image_path}'. Make sure the image is in the same folder as your runner script.")
    exit()

# 2. Preprocess the image for the model
inputs = processor(images=image, return_tensors="pt")

# 3. Run inference
with torch.no_grad():
    outputs = model(**inputs)

# 4. Post-process the results (filtering detections with a 0.5 confidence threshold)
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, threshold=0.5, target_sizes=target_sizes
)[0]

# 5. Print out what the model detected
print(f"\n--- Detection Results ---")
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    class_name = model.config.id2label[label.item()]
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected **{class_name}** with confidence **{round(score.item(), 3)}** at bounding box: {box}")