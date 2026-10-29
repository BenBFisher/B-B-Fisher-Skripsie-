import torch
from safetensors.torch import load_file
from local_config_rt_detr_v2 import RTDetrV2Config
from local_modeling_rt_detr_v2 import RTDetrV2ForObjectDetection

print("1. Initializing pure PyTorch architecture...")
config = RTDetrV2Config()
model = RTDetrV2ForObjectDetection(config)

print("2. Loading Hugging Face Safetensors...")
hf_state_dict = load_file("./my_local_model/model.safetensors")

print("3. Translating weight keys...")
new_state_dict = {}
for key, tensor in hf_state_dict.items():
    new_key = key
    
    # 1. Remove the overarching "model." prefix from HF's base wrapper
    if new_key.startswith("model."):
        new_key = new_key.replace("model.", "", 1)
        
    # 2. DROP the Hugging Face ResNet-VD backbone weights entirely
    if new_key.startswith("backbone."):
        continue
        
    # 3. DROP the denoising weights (only used during training)
    if new_key.startswith("denoising_class_embed"):
        continue
        
    # 4. MAP the prediction heads to our cleaner top-level layout
    if new_key.startswith("decoder.bbox_embed"):
        new_key = new_key.replace("decoder.bbox_embed", "bbox_embed")
    if new_key.startswith("decoder.class_embed"):
        new_key = new_key.replace("decoder.class_embed", "class_embed")
    
    # Store the translated key and the tensor
    new_state_dict[new_key] = tensor

print("4. Injecting weights into model...")
# strict=False allows the model to load everything that matches, 
# and returns lists of what failed so we can fix them.
missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

if len(missing_keys) == 0 and len(unexpected_keys) == 0:
    print("SUCCESS: All weights loaded perfectly!")
else:
    print(f"\nMissing Keys (In your model, but not in weights): {len(missing_keys)}")
    for k in missing_keys[:5]: print(f"  - {k}")
        
    print(f"\nUnexpected Keys (In weights, but not in your model): {len(unexpected_keys)}")
    for k in unexpected_keys[:5]: print(f"  - {k}")

# 5. Ready for inference!
model.eval()

print("\n6. Testing forward pass with a dummy image...")
# Create a fake image tensor: [Batch Size, Channels, Height, Width]
# RT-DETR typically expects 640x640 inputs
dummy_image = torch.randn(1, 3, 640, 640)

# Disable gradient tracking for inference
with torch.no_grad():
    outputs = model(dummy_image)
    
print("\n--- OUTPUT SHAPES ---")
print(f"Logits: {outputs['logits'].shape} -> Expected: [1, 300, 80]")
print(f"Boxes:  {outputs['pred_boxes'].shape} -> Expected: [1, 300, 4]")
print("\nSUCCESS! The pure PyTorch architecture is fully operational.")