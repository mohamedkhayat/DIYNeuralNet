import os
import numpy

_GPU_AVAILABLE = False

try:
  output = os.popen('nvidia-smi').read()
  if "NVIDIA-SMI" in output:
    _GPU_AVAILABLE = True
    import cupy
    np = cupy
    
  else:
    np = numpy
    
except Exception as e:
  np = numpy
  raise ValueError(f"Exception {e}")

def get_numpy():
  return np

def is_gpu_available():
  return _GPU_AVAILABLE