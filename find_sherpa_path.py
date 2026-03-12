"""
Find sherpa-onnx installation path
"""

import sherpa_onnx
import os

print("sherpa-onnx path:", os.path.dirname(sherpa_onnx.__file__))
print("\nFiles in sherpa-onnx directory:")
sherpa_dir = os.path.dirname(sherpa_onnx.__file__)
for file in os.listdir(sherpa_dir):
    print(f"  {file}")
