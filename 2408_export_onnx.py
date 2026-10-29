import os
import torch

# Import your custom local architecture files
from local_config_rt_detr_v2 import RTDetrV2Config
from local_modeling_rt_detr_v2 import RTDetrV2ForObjectDetection

def main():
    device = torch.device("cpu")
    print("Initializing ONNX Export on CPU...")

    # 1. Setup the 80-Class Model
    # Since we are using the original weights, num_labels must be 80
    config = RTDetrV2Config(num_labels=80)
    model = RTDetrV2ForObjectDetection(config)
    
    # 2. Load the specific COCO5k 80-Class Weights
    weights_path = "phase_1_weights_3107_coco5k.pth"
    print(f"Loading weights from: {weights_path}")
    
    if not os.path.exists(weights_path):
        print(f"Error: Could not find {weights_path}. Check the file name.")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True), strict=True)
    
    # Set model to evaluation mode (CRITICAL: turns off dropout and batch norm updates)
    model.eval()
    model.to(device)

    # 3. Create a Dummy Input Tensor
    # ONNX needs to trace a physical tensor through the network to map the graph.
    # Shape: (Batch Size, Channels, Height, Width) -> (1, 3, 640, 640)
    dummy_input = torch.randn(1, 3, 640, 640, device=device)

    # 4. Define Export Parameters
    onnx_file_path = "phase_2_rt_detr_80class_static.onnx"
    
    print("Tracing the computational graph. This may take a moment...")
    
    torch.onnx.export(
        model,                         # The loaded model
        dummy_input,                   # The dummy tracing tensor
        onnx_file_path,                # Where to save the file
        export_params=True,            # Store the trained parameter weights inside the model file
        opset_version=16,              # Opset 16 supports modern transformer operations
        do_constant_folding=True,      # Automatically optimize the graph by pre-computing constants
        input_names=['input_images'],  # Define a clear input node name
        output_names=['logits', 'pred_boxes'] # Define clear output node names
    )
    
    print("\n" + "="*45)
    print(f"SUCCESS: Model exported to {onnx_file_path}")
    print("="*45)

if __name__ == "__main__":
    main()