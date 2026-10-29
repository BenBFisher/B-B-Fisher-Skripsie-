import os
import torch

# Import your custom local architecture files
from local_config_rt_detr_v2 import RTDetrV2Config
from local_modeling_rt_detr_v2 import RTDetrV2ForObjectDetection

def main():
    device = torch.device("cpu")
    print("Initializing ONNX Export on CPU...")

    # 1. Setup the Micro 10-Class Model Configuration
    config = RTDetrV2Config(
        num_labels=2,
        num_queries=5,
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
    
    # 2. Load the specific Micro Weights
    weights_path = "micro_rtdetr_version_4full_epoch_50.pth"
    print(f"Loading weights from: {weights_path}")
    
    if not os.path.exists(weights_path):
        print(f"Error: Could not find {weights_path}. Check the file name.")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True), strict=True)
    
    # Set model to evaluation mode (CRITICAL: turns off dropout and batch norm updates)
    model.eval()
    model.to(device)

    # 3. Create a Dummy Input Tensor matched to your 160x160 footprint
    # Shape: (Batch Size, Channels, Height, Width) -> (1, 3, 160, 160)
    dummy_input = torch.randn(1, 3, 32, 32, device=device)

    # 4. Define Export Parameters
    onnx_file_path = "micro_rtdetr_32_static_4.onnx"
    
    print("Tracing the computational graph. This may take a moment...")
    
    torch.onnx.export(
        model,                         # The loaded model
        dummy_input,                   # The dummy tracing tensor
        onnx_file_path,                # Where to save the file
        export_params=True,            # Store the trained parameter weights inside the model file
        opset_version=16,              # Opset 16 supports modern transformer operations
        do_constant_folding=True,      # Automatically optimize the graph by pre-computing constants
        input_names=['input_images'],  # Define a clear input node name
        output_names=['logits', 'pred_boxes', 'aux_outputs', 'enc_outputs'], # Maps to your model's dictionary output
        dynamo=False
    )
    
    print("\n" + "="*45)
    print(f"SUCCESS: Model exported to {onnx_file_path}")
    print("="*45)

if __name__ == "__main__":
    main()