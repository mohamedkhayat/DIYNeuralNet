"""Device selection module for automatic GPU/CPU detection.

This module automatically detects if NVIDIA GPU is available through CuPy
and falls back to CPU (NumPy) if not available or if CuPy import fails.
"""

from typing import Any, Union, TYPE_CHECKING
import numpy

# Type alias for arrays that can be either numpy.ndarray or cupy.ndarray
try:
    from typing import TypeAlias

    if TYPE_CHECKING:
        try:
            import cupy

            ArrayType: TypeAlias = Union[numpy.ndarray, cupy.ndarray]
        except ImportError:
            ArrayType: TypeAlias = numpy.ndarray
    else:
        ArrayType: TypeAlias = Union[numpy.ndarray, Any]
except ImportError:
    # For Python < 3.10, we define it without TypeAlias
    if TYPE_CHECKING:
        try:
            import cupy

            ArrayType = Union[numpy.ndarray, cupy.ndarray]
        except ImportError:
            ArrayType = numpy.ndarray
    else:
        ArrayType = Union[numpy.ndarray, Any]

# Module-level variable to track GPU availability
_GPU_AVAILABLE: bool = False

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


def get_numpy() -> Union[Any, Any]:
    """Get the appropriate numpy-like module (cupy or numpy).

    Returns:
        cupy if GPU is available and detected, numpy otherwise

    Note:
        This function returns either cupy or numpy depending on GPU availability.
        Both modules have compatible APIs for most array operations.
    """
    return np


def is_gpu_available() -> bool:
    """Check if GPU is available for computation.

    Returns:
        True if NVIDIA GPU is available and CuPy is working, False otherwise
    """
    return _GPU_AVAILABLE
