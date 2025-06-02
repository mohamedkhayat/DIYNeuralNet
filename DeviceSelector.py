# I really should make this a class
import numpy

_GPU_AVAILABLE = False

try:
    import cupy

    if cupy.cuda.is_available():
        _GPU_AVAILABLE = True
        np = cupy
        print("Using GPU")
    else:
        print("No Available GPU Detected")
        print("Falling back to CPU")
        np = numpy

except ImportError as e:
    print(f"Error when importing cupy : {e}")
    print("Falling back to CPU")
    np = numpy

except Exception as e:
    print(f"Error : {e}")
    print("Falling back to CPU")
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
