import os
from ai_edge_quantizer import quantizer, recipe

# Check your savedmodel folder to confirm this is the exact name onnx2tf generated
INPUT_TFLITE = "micro_rtdetr_32_static_1_2_fake_savedmodel/micro_rtdetr_32_static_1_2_fake_float32.tflite" 

# Name this exactly what compile_microtvm.py is looking for
OUTPUT_TFLITE = "micro_rtdetr_32.tflite"

def main():
    if not os.path.exists(INPUT_TFLITE):
        print(f"Error: Could not find input file: {INPUT_TFLITE}")
        return

    print(f"Loading model: {INPUT_TFLITE}")
    qt = quantizer.Quantizer(INPUT_TFLITE)
    
    print("Applying Dynamic INT8 Recipe (wi8_afp32)...")
    qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
    
    print("Quantizing and exporting INT8 model...")
    qt.quantize().export_model(OUTPUT_TFLITE)
    
    file_size_mb = os.path.getsize(OUTPUT_TFLITE) / (1024 * 1024)
    print("\n" + "=" * 45)
    print(f"SUCCESS: Generated {OUTPUT_TFLITE}")
    print(f"Quantized Model Size: {file_size_mb:.2f} MB")
    print("=" * 45)

if __name__ == "__main__":
    main()