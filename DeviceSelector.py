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
  """
  If nvidia gpu is detected, np == cupy, an alternative to numpy that uses the GPU to accelerate
  computation
  """
  return np

def is_gpu_available():
  """
  returns True if gpu is available, False if not
  """
  return _GPU_AVAILABLE