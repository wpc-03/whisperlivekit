"""
Find where generate_keywords is located in sherpa-onnx
"""

import sherpa_onnx

print("sherpa-onnx attributes:")
for item in dir(sherpa_onnx):
    if 'keyword' in item.lower() or 'generate' in item.lower():
        print(f"  {item}")

# Also check keyword_spotter module
print("\n--- Checking keyword_spotter ---")
if hasattr(sherpa_onnx, 'keyword_spotter'):
    print(dir(sherpa_onnx.keyword_spotter))
