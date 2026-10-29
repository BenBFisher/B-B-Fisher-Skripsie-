import time
import numpy as np
import onnxruntime as ort

def main():
    #onnx_model_path = "phase_2_rt_detr_80class_static.onnx"
    onnx_model_path = "phase_2_rt_detr_80class_int8.onnx"
    
    # 1. Initialize the Execution Session
    # Use 'CPUExecutionProvider' for CPU testing or 'CUDAExecutionProvider' for GPU
    providers = ['CPUExecutionProvider']
    print(f"Loading {onnx_model_path} with {providers[0]}...")
    

    # Update your session options to force maximum graph optimization
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(onnx_model_path, sess_options, providers=providers)
    #till here
    #session = ort.InferenceSession(onnx_model_path, providers=providers)
    
    # 2. Inspect Model Input and Output Nodes
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Input Node: {input_name} | Expected Shape: {input_shape}")
    
    # 3. Create a Dummy Test Image (1, 3, 640, 640) in Float32
    # In practice, replace this with a preprocessed, normalized NumPy array of a real image
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    
    # 4. Warm-Up Runs (To initialize memory pools and avoid skewing metrics)
    print("Warming up runtime engine...")
    for _ in range(5):
        _ = session.run(None, {input_name: dummy_input})
        
    # 5. Benchmark Latency
    num_runs = 50
    print(f"Benchmarking inference over {num_runs} iterations...")
    
    start_time = time.perf_counter()
    for _ in range(num_runs):
        outputs = session.run(None, {input_name: dummy_input})
    total_time = time.perf_counter() - start_time
    
    avg_latency_ms = (total_time / num_runs) * 1000
    fps = num_runs / total_time
    
    # 6. Display Performance Metrics
    print("\n" + "=" * 45)
    print("=== ONNX RUNTIME PERFORMANCE METRICS ===")
    print("=" * 45)
    print(f"Average Latency: {avg_latency_ms:.2f} ms")
    print(f"Throughput:      {fps:.2f} FPS")
    print("=" * 45)
    
    # outputs[0] -> logits (1, 300, 80)
    # outputs[1] -> pred_boxes (1, 300, 4)
    print(f"Output 0 (Logits) Shape:     {outputs[0].shape}")
    print(f"Output 1 (Boxes) Shape:      {outputs[1].shape}")

if __name__ == "__main__":
    main()