import cv2
import time
import numpy as np
import tensorflow as tf
import psutil
import os

def benchmark_video_file(video_path="Test_video.mp4", model_path="phase_2_rt_detr_80class_int8.tflite"):
    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=8)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    input_index = input_details['index']
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    infer_times = []
    
    print("Benchmarking sequential video frames...")
    
    while True:
        status, frame = cap.read()
        if not status:
            print("End of video file reached.")
            break  # Exit loop when video ends
            
        t0 = time.perf_counter()
        
        # Preprocessing
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (640, 640))
        input_tensor = ((resized / 255.0) - mean) / std
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(input_details['dtype'])
        
        # Inference
        interpreter.set_tensor(input_index, input_tensor)
        interpreter.invoke()
        
        t1 = time.perf_counter()
        infer_times.append((t1 - t0) * 1000)

    cap.release()
    
    if infer_times:
        print(f"\nProcessed {len(infer_times)} frames.")
        print(f"Average Total Latency: {np.mean(infer_times):.2f} ms")
        print(f"Effective Throughput:  {1000.0 / np.mean(infer_times):.2f} FPS")

if __name__ == "__main__":
    benchmark_video_file()