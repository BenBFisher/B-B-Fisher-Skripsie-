import os
import numpy as np
import tensorflow as tf
from ai_edge_quantizer import calibrator, quantizer

INPUT_TFLITE = "mirco_rt_detr_32_tf_fake_savedmodel/micro_rtdetr_32_static_1_fake_float32.tflite"
OUTPUT_TFLITE = "micro_rtdetr_32_3_1_fake_int8.tflite"

def main():
    if not os.path.exists(INPUT_TFLITE):
        print(f"Error: Could not find input file: {INPUT_TFLITE}")
        return

    # Dynamically grab your model's specific input tensor name 
    interpreter = tf.lite.Interpreter(model_path=INPUT_TFLITE)
    input_details = interpreter.get_input_details()
    input_name = input_details[0]['name']
    
    print(f"Loading Float32 model: {INPUT_TFLITE}")
    qt = quantizer.Quantizer(INPUT_TFLITE)
    
    # "static_wi8_ai8" natively forces both internal layers AND input/outputs to INT8
    print("Applying Static Full INT8 Recipe...")
    qt.load_quantization_recipe("static_wi8_ai8")
    
    print(f"Generating calibration dataset for tensor: '{input_name}'...")
    # AI Edge Quantizer requires calibration data structured as a dictionary of signatures
    calibration_data = {
        "serving_default": [
            {input_name: np.random.rand(1, 32, 32, 3).astype(np.float32)} for _ in range(100)
        ]
    }
    
    print("Running C++ Profiler Calibration (this may take a minute)...")
    calibration_result = qt.calibrate(
        calibration_data,
        mode=calibrator.CalibrationMode.CALIBRATION_PROFILER_BASED,
    )
    
    print("Quantizing and exporting INT8 model...")
    qt.quantize(calibration_result=calibration_result).export_model(OUTPUT_TFLITE)
    
    file_size_mb = os.path.getsize(OUTPUT_TFLITE) / (1024 * 1024)
    print("\n" + "=" * 45)
    print(f"SUCCESS: Generated {OUTPUT_TFLITE}")
    print(f"Quantized Model Size: {file_size_mb:.2f} MB")
    print("=" * 45)

if __name__ == "__main__":
    main()