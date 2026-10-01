"""
quantize_models.py
==================
Quantizes FP32 ONNX RapidOCR models into INT8 dynamic quantized models.
Reduces file size dramatically (~60-75% smaller) while preserving high OCR accuracy and accelerating CPU inference.
"""

import os
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_rapidocr_models():
    base_dir = Path(__file__).parent / "models" / "rapidocr_multilingual"
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist.")
        return

    print("==================================================")
    print("      RAPIDOCR ONNX MODEL INT8 QUANTIZATION      ")
    print("==================================================")

    for lang_dir in base_dir.iterdir():
        if not lang_dir.is_dir():
            continue

        rec_onnx = lang_dir / "rec.onnx"
        if not rec_onnx.exists():
            continue

        quant_onnx = lang_dir / "rec_quant.onnx"
        print(f"\n[+] Processing {lang_dir.name.upper()}...")
        print(f"    Input:  {rec_onnx.name}")

        try:
            quantize_dynamic(
                model_input=str(rec_onnx),
                model_output=str(quant_onnx),
                weight_type=QuantType.QUInt8
            )
            orig_size = rec_onnx.stat().st_size / (1024 * 1024)
            quant_size = quant_onnx.stat().st_size / (1024 * 1024)
            reduction = ((orig_size - quant_size) / orig_size) * 100
            print(f"    Output: {quant_onnx.name}")
            print(f"    Original Size:  {orig_size:.2f} MB")
            print(f"    Quantized Size: {quant_size:.2f} MB (Reduced by {reduction:.1f}%)")
        except Exception as e:
            print(f"    [-] Quantization failed for {lang_dir.name}: {e}")

    print("\n[+] Quantization complete!")


if __name__ == "__main__":
    quantize_rapidocr_models()
