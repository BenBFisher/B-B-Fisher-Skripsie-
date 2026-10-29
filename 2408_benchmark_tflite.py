import time
import numpy as np
import tensorflow as tf

def main():
    # Update this path if the file is inside a subfolder
    tflite_model_path = "phase_2_rt_detr_80class_int8.tflite"
    
    print(f"Loading TFLite model from {tflite_model_path}...")
    
    # 1. Initialize the Interprete      
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path, num_threads=16)
    
    # 2. Allocate memory for the model's tensors
    interpreter.allocate_tensors()
    
    # 3. Get Input and Output Node details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    print(f"Input Node Expected Shape: {input_shape}")
    
    # 4. Create a Dummy Test Image
    # TFLite expects standard Float32 inputs; the engine handles the FP16 conversion internally
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # 5. Warm-up Runs
    print("Warming up TFLite engine...")
    for _ in range(5):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        
    # 6. Benchmark Latency
    num_runs = 50
    print(f"Benchmarking inference over {num_runs} iterations...")
    
    start_time = time.perf_counter()
    for _ in range(num_runs):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        # Extract outputs (Optional, but ensures the graph fully executes)
        _ = interpreter.get_tensor(output_details[0]['index'])
    total_time = time.perf_counter() - start_time
    
    avg_latency_ms = (total_time / num_runs) * 1000
    fps = num_runs / total_time
    
    print("\n" + "=" * 45)
    print("=== TFLITE FP16 PERFORMANCE METRICS ===")
    print("=" * 45)
    print(f"Average Latency: {avg_latency_ms:.2f} ms")
    print(f"Throughput:      {fps:.2f} FPS")
    print("=" * 45)

if __name__ == "__main__":
    main()