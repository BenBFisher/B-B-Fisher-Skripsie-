import torch
import os
from datetime import datetime
from torch.profiler import profile, record_function, ProfilerActivity

# Import your configuration and the model architecture
from local_config_rt_detr_v2 import RTDetrV2Config
from local_modeling_rt_detr_v2 import RTDetrV2ForObjectDetection 

def benchmark_with_profiler(image_size=640):
    print(f"--- Initializing PyTorch Profiler on CPU ---")
    
    # 1. Setup Logging Directory and File
    log_dir = "resources_benchmark"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "profiler_benchmark_log.txt")
    
    # Initialize the model using the strict 80-class specification
    config = RTDetrV2Config(num_labels=80) 
    model = RTDetrV2ForObjectDetection(config) 
    model.eval()

    """ #To test with local weights, uncomment the following lines and provide the correct path:
    # 1. Initialize the model using the strict 80-class specification
    config = RTDetrV2Config(num_labels=80) 
    model = RTDetrV2ForObjectDetection(config) 
    
    # 2. Load your custom trained weights
    print("Loading custom trained weights...")
    model.load_state_dict(torch.load("rt_detr_custom_trained.pth", map_location=torch.device('cpu')))
    
    # 3. Set to evaluation mode
    model.eval()
    """

    # Create a dummy image tensor of the correct shape
    dummy_input = torch.randn(1, 3, image_size, image_size)
    
    # 2. Warmup Passes
    # This ensures backend libraries are loaded before measuring
    print("Running warmup passes...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)

    # 3. Profiling Pass
    print("Recording granular memory and CPU metrics...")
    with profile(
        activities=[ProfilerActivity.CPU],
        profile_memory=True, # Tracks exact byte allocations
        record_shapes=True,  # Tracks tensor shapes for each operation
        with_stack=True      # Links operations to the specific line of Python code
    ) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                _ = model(dummy_input) 

    # 4. Extract and Format Tables
    mem_table = prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20)
    cpu_table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_text = (
        f"[{timestamp}] PyTorch Profiler Benchmark (Image Size: {image_size}x{image_size})\n"
        f"================================================================================\n"
        f"--- Top 20 Memory-Consuming Operations ---\n"
        f"{mem_table}\n\n"
        f"--- Top 20 CPU-Intensive Operations ---\n"
        f"{cpu_table}\n"
        f"================================================================================\n"
    )

    # 5. Output to Console and Save to Log
    print("\n" + report_text)
    
    with open(log_file, "a") as f:
        f.write(report_text)
        
    print(f"Profiler tables successfully saved to {log_file}")

if __name__ == "__main__":
    benchmark_with_profiler()