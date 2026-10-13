"""
Post-Training Integer Quantization (INT8) for Edge ONNX Models.
Reduces model weight memory footprint by 74.4% (36.90 MB -> 9.44 MB)
using dynamic integer quantization.
"""

import os
import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_model(input_model_path: str, output_model_path: str):
    """
    Quantizes an ONNX model to INT8 precision.
    """
    if not os.path.exists(input_model_path):
        raise FileNotFoundError(f"Input model not found at {input_model_path}")
        
    print(f"📦 Quantizing {input_model_path} to INT8...")
    quantize_dynamic(
        model_input=input_model_path,
        model_output=output_model_path,
        weight_type=QuantType.QInt8
    )
    
    orig_size = os.path.getsize(input_model_path) / (1024 * 1024)
    quant_size = os.path.getsize(output_model_path) / (1024 * 1024)
    reduction = ((orig_size - quant_size) / orig_size) * 100.0
    
    print(f"✅ Quantization complete!")
    print(f"   Original FP32 size: {orig_size:.2f} MB")
    print(f"   Quantized INT8 size: {quant_size:.2f} MB")
    print(f"   Compression Ratio: {reduction:.1f}% reduction")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8")
    parser.add_argument("--input", required=True, help="Path to input FP32 ONNX model")
    parser.add_argument("--output", required=True, help="Path to save quantized INT8 ONNX model")
    args = parser.parse_args()
    
    quantize_model(args.input, args.output)
