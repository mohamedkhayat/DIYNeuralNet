import os
import numpy
import cupy
_GPU_AVAILABLE = False

try:
  output = os.popen('nvidia-smi').read()
  if "NVIDIA-SMI" in output:
    try:
      print(cupy.cuda.runtime.getDeviceCount())
      _GPU_AVAILABLE = True
      np = cupy
    except Exception as e:
      np = numpy
  else:
    np = numpy
except Exception as e:
  np = numpy
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