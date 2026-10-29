import torch
import psutil
import os
from datetime import datetime
from torch.utils.benchmark import Timer
from local_config_rt_detr_v2 import RTDetrV2Config
from local_modeling_rt_detr_v2 import RTDetrV2ForObjectDetection

def benchmark_resources(device="cpu", image_size=640, min_run_time=15.0):
    print(f"--- Benchmarking Deep Resources on {device.upper()} ---")
    
    # 1. Setup Logging Directory and File
    log_dir = "resources_benchmark"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "cpu_benchmark_log.txt")
    
    process = psutil.Process()
    ram_baseline_mb = process.memory_info().rss / (1024 ** 2)
    
    # 2. Initialize Model (strictly maintaining 80 classes)
    config = RTDetrV2Config(num_labels=80) 
    model = RTDetrV2ForObjectDetection(config).to(device)
    model.eval()

    #To test with local weights, uncomment the following lines and provide the correct path:
    # 1. Initialize the model using the strict 80-class specification
    config = RTDetrV2Config(num_labels=80) 
    model = RTDetrV2ForObjectDetection(config) 
    
    # 2. Load your custom trained weights
    print("Loading custom trained weights...")
    model.load_state_dict(torch.load("phase_1_weights_3107_cocofull.pth", map_location=torch.device('cpu')))
    
    # 3. Set to evaluation mode
    model.eval()
    
    
    ram_after_model_mb = process.memory_info().rss / (1024 ** 2)
    model_ram_mb = ram_after_model_mb - ram_baseline_mb
    
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    
    # 3. Setup CPU Measurement Baselines
    _ = process.cpu_percent(interval=None) # Initialize percentage tracking
    cpu_times_before = process.cpu_times() 
    ctx_switches_before = process.num_ctx_switches()

    # 4. Latency & Throughput (The Core Load)
    print(f"Running extended benchmark for ~{min_run_time} seconds...")
    timer = Timer(
        stmt="model(x)",
        globals={"model": model, "x": dummy_input},
        num_threads=1 # Pin to a single thread
    )
    
    result = timer.blocked_autorange(min_run_time=min_run_time) 
    latency_ms = result.median * 1000
    fps = 1.0 / result.median
    
    # 5. Measure Final CPU Metrics
    cpu_usage = process.cpu_percent(interval=None)
    cpu_times_after = process.cpu_times()
    ctx_switches_after = process.num_ctx_switches()
    
    # Calculate Process-Specific Times
    # User time represents time spent by normal processes executing in user mode
    user_time_sec = cpu_times_after.user - cpu_times_before.user 
    # System time represents time spent by processes executing in kernel mode
    system_time_sec = cpu_times_after.system - cpu_times_before.system 
    
    # Calculate Context Switches
    vol_ctx = ctx_switches_after.voluntary - ctx_switches_before.voluntary
    invol_ctx = ctx_switches_after.involuntary - ctx_switches_before.involuntary
    
    # Calculate Core Architecture
    # Counts physical cores, excluding hyper-threading
    physical_cores = psutil.cpu_count(logical=False) 
    # Counts total logical threads
    logical_cores = psutil.cpu_count(logical=True) 
    
    # Calculate Frequency 
    cpu_freq = psutil.cpu_freq()
    freq_str = f"{cpu_freq.current:.2f} MHz (Min: {cpu_freq.min} / Max: {cpu_freq.max})" if cpu_freq else "N/A"
    
    # Memory Info
    mem_info = process.memory_info()
    total_ram_mb = mem_info.rss / (1024 ** 2)
    peak_ram_mb = getattr(mem_info, 'peak_wset', mem_info.rss) / (1024 ** 2)
    
    # Safely Check Hardware Temperature
    temp_str = "N/A (Not natively supported on this OS/Hardware without WMI)"
    if hasattr(psutil, "sensors_temperatures"):
        temps = psutil.sensors_temperatures()
        if temps:
            # Grab the first available temperature sensor dict
            first_sensor = list(temps.values())[0]
            if first_sensor:
                temp_str = f"{first_sensor[0].current}°C"

    # 6. Format the Results
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_output = (
        f"[{timestamp}] Device: {device.upper()} | Image Size: {image_size}x{image_size}\n"
        f"Total Model Parameters: {num_params:,} (Classes: {model.config.num_labels})\n"
        f"--- Memory Profile ---\n"
        f"  Model Weights Memory:    {model_ram_mb:.2f} MB\n"
        f"  Total Process RAM (RSS): {total_ram_mb:.2f} MB\n"
        f"  Peak RAM Footprint:      {peak_ram_mb:.2f} MB\n"
        f"--- CPU Architecture ---\n"
        f"  Available Cores:         {physical_cores} Physical / {logical_cores} Logical\n"
        f"  Clock Frequency:         {freq_str}\n"
        f"  Die Temperature:         {temp_str}\n"
        f"--- CPU Execution Profile ---\n"
        f"  Average Process Usage:   {cpu_usage:.2f}%\n"
        f"  Execution Time (User):   {user_time_sec:.2f} seconds\n"
        f"  Kernel Time (System):    {system_time_sec:.2f} seconds\n"
        f"  Context Switches:        {vol_ctx} Voluntary / {invol_ctx} Involuntary\n"
        f"--- Throughput Results ---\n"
        f"  Median Latency:          {latency_ms:.2f} ms\n"
        f"  Throughput:              {fps:.2f} FPS\n"
        f"{'-'*60}\n"
    )
    
    # 7. Output to Console and Save to Log
    print("\n--- Final Results ---")
    print(log_output)
    
    with open(log_file, "a") as f:
        f.write(log_output)
        
    print(f"Results successfully saved to {log_file}")

if __name__ == "__main__":
    benchmark_resources(device="cpu")