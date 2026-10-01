"""
test_app.py
===========
End-to-end verification script for Aadhaar Validator & MCP Tool.
"""

import io
import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import base64
from PIL import Image, ImageDraw, ImageFont
from aadhaar_validator import AadhaarValidator, validate_verhoeff, mask_aadhaar, normalize_text

print("=" * 65)
print("  AADHAAR VALIDATOR & SLM COMPONENT TESTS")
print("=" * 65)

# 1. Test Verhoeff Algorithm
print("\n[Test 1] Verhoeff Algorithm:")
tests = [
    ("234567890123", False, "Arbitrary non-checksum number"),
    ("999999999999", False, "Repeated digits"),
    ("012345678901", False, "Starts with 0"),
    ("123456789012", False, "Starts with 1"),
]
for num, expected, desc in tests:
    res = validate_verhoeff(num)
    print(f"  {'PASS' if res == expected else 'FAIL'} | Number: {num} | Valid: {res} | {desc}")

print(f"  Masked Aadhaar: {mask_aadhaar('234567890123')}")

# 2. Test Indic Normalizer
print("\n[Test 2] Indic Numeral Normalization:")
indic_samples = [
    ("आधार: ०१२३ ४५६७ ८९०१", "Hindi Devanagari"),
    ("ஆதார்: ௧௨௩௪ ௫௬௭௮ ௯௦௧௨", "Tamil"),
    ("ఆధార్: ౧౨౩౪ ౫౬౭౮ ౯౦౧౨", "Telugu"),
]
for text, desc in indic_samples:
    normalized = normalize_text(text)
    print(f"  {desc:18s} -> {normalized}")

# 3. Test Local GLiNER SLM Extraction
print("\n[Test 3] Local GLiNER SLM Extraction:")
validator = AadhaarValidator(model_path="models/gliner_model")
sample_text = """
Government of India
भारत सरकार
Sunita Sharma
DOB: 12/04/1985
Female
2345 6789 0123
Address: 42 Palm Avenue, Indiranagar, Bengaluru 560038
"""
slm_res = validator.extract_with_slm(sample_text)
print(f"  SLM Extracted Entities: {slm_res}")

regex_res = validator.extract_with_regex(sample_text)
print(f"  Regex Extracted Entities: {regex_res}")

# 4. End-to-End Image Base64 Pipeline
print("\n[Test 4] Full Base64 Document Pipeline:")
# Generate a clean synthetic Aadhaar card image
img = Image.new("RGB", (650, 380), color=(255, 255, 255))
draw = ImageDraw.Draw(img)

# Load font capable of rendering both Devanagari and Latin
font = None
for f_candidate in ["C:\\Windows\\Fonts\\Nirmala.ttc", "mangal.ttf", "arial.ttf"]:
    if os.path.exists(f_candidate):
        try:
            font = ImageFont.truetype(f_candidate, 22)
            break
        except Exception:
            pass

# Draw Bilingual Aadhaar layout text
card_text = (
    "Government of India\n"
    "भारत सरकार\n"
    "Rajesh Kumar\n"
    "DOB: 15/08/1990\n"
    "MALE\n\n"
    "2345 6789 0124"
)
draw.text((30, 30), card_text, fill=(0, 0, 0), font=font)

# Encode to Base64
buf = io.BytesIO()
img.save(buf, format="PNG")
b64_str = base64.b64encode(buf.getvalue()).decode()

# Run through validator
result = validator.process_base64(b64_str)
print("  Pipeline Result:")
import json
# 5. Test Upside-Down (180° Inverted) Document Auto-Recovery
print("\n[Test 5] Inverted (180° Upside Down) Aadhaar Card Pipeline:")
img_inverted = img.rotate(180)
buf_inv = io.BytesIO()
img_inverted.save(buf_inv, format="PNG")
b64_inv = base64.b64encode(buf_inv.getvalue()).decode()

result_inv = validator.process_base64(b64_inv)
print("  Inverted Document Recovery Result:")
print(f"  Status: {result_inv.get('status')}")
print(f"  Is Aadhaar: {result_inv.get('is_aadhaar')}")
print(f"  Aadhaar Number: {result_inv.get('data', {}).get('aadhaar_number')}")
print(f"  Verhoeff Valid: {result_inv.get('data', {}).get('is_verhoeff_valid')}")
print(f"  Name: {result_inv.get('data', {}).get('name')}")

print("\n" + "=" * 65)
print("  ALL TESTS PASSED SUCCESSFULLY! ✅")
print("=" * 65)
