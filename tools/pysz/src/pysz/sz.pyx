# distutils: language = c++
"""Python interface for SZ3 compression library."""

from pysz cimport sz as c_sz
cimport cython
from cython.operator cimport dereference
import numpy as np
cimport numpy as cnp
from libc.stdint cimport int8_t, int32_t, int64_t
from libc.stddef cimport size_t
from libc.string cimport memcpy
from typing import Tuple
from libcpp.vector cimport vector
from libcpp.string cimport string

# Initialize NumPy C API
cnp.import_array()


class szErrorBoundMode:
    """Error bound modes"""
    ABS = 0
    REL = 1
    PSNR = 2
    L2NORM = 3
    ABS_AND_REL = 4
    ABS_OR_REL = 5


class szAlgorithm:
    """Compression algorithms"""
    LORENZO_REG = 0
    INTERP_LORENZO = 1
    INTERP = 2
    NOPRED = 3
    LOSSLESS = 4


cdef class szConfig:
    """
    Configuration class for SZ3 compression.
    
    Enum classes: szErrorBoundMode, szAlgorithm (module level)
    
    Parameters: Dimension sizes as individual ints, tuple, or list
        szConfig(data.shape), szConfig(100, 200, 300), etc.
    """

    def __init__(self, *args):
        """Initialize config with optional dimensions."""
        self.conf = Config()
        self._pred_idx_array = None
        self._anchor_idx_array = None
        self._anchor_value_array = None
        self._block_sizes_array = None
        if args:
            self.setDims(*args)
    
    def setDims(self, *args):
        """Set dimensions. Accepts tuple/list or individual integers."""
        cdef vector[size_t] dims
        
        # Handle both setDims(100, 200, 300) and setDims((100, 200, 300))
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            dims_iter = args[0]
        else:
            dims_iter = args
        
        if not dims_iter:
            raise ValueError("At least one dimension required")
        
        for arg in dims_iter:
            if not isinstance(arg, int) or arg <= 0:
                raise ValueError(f"Dimension must be positive integer, got {arg}")
            dims.push_back(<size_t>arg)
        
        self.conf.setDims(dims.begin(), dims.end())

    def loadcfg(self, cfgpath: str):
        """Load configuration from INI file."""
        cdef string cfgpathStr = cfgpath.encode('utf-8')
        try:
            self.conf.loadcfg(cfgpathStr)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to load '{cfgpath}': {e}")
    
    # Read-only properties
    @property
    def dims(self):
        """Get dimensions (read-only)."""
        return tuple(self.conf.dims)
    
    @property
    def num_elements(self):
        """Get total number of elements (read-only)."""
        return self.conf.num
    
    @property
    def ndim(self):
        """Get number of dimensions (read-only)."""
        return self.conf.N
    
    # Error bounds (double)
    @property
    def absErrorBound(self):
        return self.conf.absErrorBound
    
    @absErrorBound.setter
    def absErrorBound(self, double value):
        self.conf.absErrorBound = value
    
    @property
    def relErrorBound(self):
        return self.conf.relErrorBound
    
    @relErrorBound.setter
    def relErrorBound(self, double value):
        self.conf.relErrorBound = value
    
    @property
    def psnrErrorBound(self):
        return self.conf.psnrErrorBound
    
    @psnrErrorBound.setter
    def psnrErrorBound(self, double value):
        self.conf.psnrErrorBound = value
    
    @property
    def l2normErrorBound(self):
        return self.conf.l2normErrorBound
    
    @l2normErrorBound.setter
    def l2normErrorBound(self, double value):
        self.conf.l2normErrorBound = value

    @property
    def quant_inds_entropy(self):
        return self.conf.quant_inds_entropy
    
    # Enum properties (accept enum or int)
    @property
    def errorBoundMode(self):
        """Get/set error bound mode. Use szErrorBoundMode enum."""
        return self.conf.errorBoundMode
    
    @errorBoundMode.setter
    def errorBoundMode(self, value):
        if isinstance(value, int):
            self.conf.errorBoundMode = value
        elif hasattr(value, 'value'):  # Is an enum
            self.conf.errorBoundMode = value.value
        else:
            raise TypeError(f"Expected int or Enum, got {type(value).__name__}")
    
    @property
    def cmprAlgo(self):
        """Get/set compression algorithm. Use szAlgorithm enum."""
        return self.conf.cmprAlgo
    
    @cmprAlgo.setter
    def cmprAlgo(self, value):
        if isinstance(value, int):
            self.conf.cmprAlgo = value
        elif hasattr(value, 'value'):  # Is an enum
            self.conf.cmprAlgo = value.value
        else:
            raise TypeError(f"Expected int or Enum, got {type(value).__name__}")
    
    @property
    def openmp(self):
        return self.conf.openmp
    
    @openmp.setter
    def openmp(self, bint value):
        self.conf.openmp = value

    @property
    def useGrandparentPredictor(self):
        return self.conf.useGrandparentPredictor

    @useGrandparentPredictor.setter
    def useGrandparentPredictor(self, bint value):
        self.conf.useGrandparentPredictor = value

    @property
    def grandparentPredictorRatio(self):
        return self.conf.grandparentPredictorRatio

    @grandparentPredictorRatio.setter
    def grandparentPredictorRatio(self, double value):
        self.conf.grandparentPredictorRatio = value

    @property
    def blockSizes(self):
        return tuple(self.conf.blockSizes)

    @blockSizes.setter
    def blockSizes(self, value):
        cdef vector[size_t] block_sizes
        if value is None:
            self._block_sizes_array = None
            self.conf.blockSizes.clear()
            return

        block_sizes_iter = tuple(value)
        for entry in block_sizes_iter:
            if not isinstance(entry, int) or entry <= 0:
                raise ValueError(f"Block size must be a positive integer, got {entry}")
            block_sizes.push_back(<size_t>entry)
        self._block_sizes_array = block_sizes_iter
        self.conf.blockSizes = block_sizes

    @property
    def quantIndsPath(self):
        if self.conf.quantIndsPath.empty():
            return None
        return (<bytes>self.conf.quantIndsPath).decode('utf-8')

    @quantIndsPath.setter
    def quantIndsPath(self, value):
        cdef string quant_inds_path
        if value is None:
            self.conf.quantIndsPath = string()
            return
        if not isinstance(value, str):
            raise TypeError(f"Expected str or None, got {type(value).__name__}")
        quant_inds_path = value.encode('utf-8')
        self.conf.quantIndsPath = quant_inds_path

    @property
    def predIdx(self):
        return self._pred_idx_array

    @predIdx.setter
    def predIdx(self, value):
        cdef cnp.ndarray[cnp.int64_t, ndim=1] pred_idx_array
        if value is None:
            self._pred_idx_array = None
            self.conf.predIdx = <const int64_t*>NULL
            self.conf.predIdxSize = 0
            return

        pred_idx_array = np.ascontiguousarray(value, dtype=np.int64)
        self._pred_idx_array = pred_idx_array
        self.conf.predIdx = <const int64_t*>cnp.PyArray_DATA(pred_idx_array)
        self.conf.predIdxSize = <size_t>pred_idx_array.size

    @property
    def predIdxSize(self):
        return self.conf.predIdxSize

    @property
    def anchorIdx(self):
        return self._anchor_idx_array

    @anchorIdx.setter
    def anchorIdx(self, value):
        cdef cnp.ndarray[cnp.int64_t, ndim=1] anchor_idx_array
        if value is None:
            self._anchor_idx_array = None
            self.conf.anchorIdx = <const int64_t*>NULL
            self.conf.anchorIdxSize = 0
            return

        anchor_idx_array = np.ascontiguousarray(value, dtype=np.int64)
        self._anchor_idx_array = anchor_idx_array
        self.conf.anchorIdx = <const int64_t*>cnp.PyArray_DATA(anchor_idx_array)
        self.conf.anchorIdxSize = <size_t>anchor_idx_array.size

    @property
    def anchorIdxSize(self):
        return self.conf.anchorIdxSize

    @property
    def anchorValue(self):
        return self._anchor_value_array

    @anchorValue.setter
    def anchorValue(self, value):
        cdef cnp.ndarray[cnp.double_t, ndim=1] anchor_value_array
        if value is None:
            self._anchor_value_array = None
            self.conf.anchorValue = <const double*>NULL
            self.conf.anchorValueSize = 0
            return

        anchor_value_array = np.ascontiguousarray(value, dtype=np.float64)
        self._anchor_value_array = anchor_value_array
        self.conf.anchorValue = <const double*>cnp.PyArray_DATA(anchor_value_array)
        self.conf.anchorValueSize = <size_t>anchor_value_array.size

    @property
    def anchorValueSize(self):
        return self.conf.anchorValueSize

    def __repr__(self):
        return f"szConfig(dims={self.dims}, num_elements={self.num_elements})"


cdef class sz:
    """SZ3 compression/decompression with zero-copy NumPy API."""
    
    # Supported dtypes
    _SUPPORTED_DTYPES = frozenset([
        np.dtype(np.float32),
        np.dtype(np.float64),
        np.dtype(np.int32),
        np.dtype(np.int64),
    ])
    
    @staticmethod
    def compress(cnp.ndarray data, config) -> Tuple[cnp.ndarray, float]:
        """
        Compress NumPy array using SZ3.
        
        Parameters
        ----------
        data : numpy.ndarray
            Input data (float32, float64, int32, or int64)
        config : szConfig
            Configuration object
        
        Returns
        -------
        compressed : numpy.ndarray
            Compressed data as uint8 array
        ratio : float
            Compression ratio (original_size / compressed_size)
        """
        # Validate dtype
        if data.dtype not in sz._SUPPORTED_DTYPES:
            raise TypeError(
                f"Unsupported dtype: {data.dtype}. "
                f"Supported: float32, float64, int32, int64"
            )
        
        # Handle config parameter
        # We need to cast the python object to our cdef class to access C++ internals
        cdef szConfig conf
        if isinstance(config, szConfig):
            conf = config
            # Always set dimensions based on input data
            shape_tuple = tuple(<Py_ssize_t>data.shape[i] for i in range(data.ndim))
            conf.setDims(*shape_tuple)
        else:
            raise TypeError(f"config must be szConfig, got {type(config)}")
        
        # Ensure C-contiguous for zero-copy
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        
        # Get data pointer (zero-copy input!)
        cdef void* data_ptr = <void*> cnp.PyArray_DATA(data)
        cdef size_t original_size = data.nbytes
        
        # Allocate buffer for compressed data (2x original size to be safe)
        cdef size_t buffer_size = <size_t>(original_size * 2)
        cdef cnp.ndarray[cnp.uint8_t, ndim=1] compressed = np.empty(buffer_size, dtype=np.uint8)
        cdef char* compressed_ptr = <char*> cnp.PyArray_DATA(compressed)
        cdef size_t compressed_size = 0
        
        # Compress into pre-allocated buffer
        if data.dtype == np.float32:
            compressed_size = c_sz.SZ_compress[float](
                conf.conf,
                <float*>data_ptr,
                compressed_ptr,
                buffer_size
            )
        elif data.dtype == np.float64:
            compressed_size = c_sz.SZ_compress[double](
                conf.conf,
                <double*>data_ptr,
                compressed_ptr,
                buffer_size
            )
        elif data.dtype == np.int32:
            compressed_size = c_sz.SZ_compress[int32_t](
                conf.conf,
                <int32_t*>data_ptr,
                compressed_ptr,
                buffer_size
            )
        elif data.dtype == np.int64:
            compressed_size = c_sz.SZ_compress[int64_t](
                conf.conf,
                <int64_t*>data_ptr,
                compressed_ptr,
                buffer_size
            )
        
        if compressed_size == 0:
            raise RuntimeError("Compression failed")
        
        # Resize to actual compressed size (no copy, just view)
        compressed = compressed[:compressed_size]
        
        ratio = original_size / float(compressed_size)
        return compressed, ratio
    

    @staticmethod
    def decompress(cnp.ndarray compressed, dtype, shape) -> Tuple[cnp.ndarray, szConfig]:
        """
        Decompress SZ3-compressed data.
        
        Parameters
        ----------
        compressed : numpy.ndarray
            Compressed data (uint8 array)
        dtype : numpy.dtype or type
            Data type of original data
        shape : tuple
            Shape of the original data
        
        Returns
        -------
        decompressed : numpy.ndarray
            Decompressed data with the specified shape
        config : szConfig
            Configuration object used during decompression (contains actual compression params)
        """
        # Validate compressed data
        if compressed.dtype != np.uint8:
            raise TypeError(f"Compressed data must be uint8, got {compressed.dtype}")
        
        # Validate and normalize dtype
        dtype = np.dtype(dtype)
        if dtype not in sz._SUPPORTED_DTYPES:
            raise TypeError(
                f"Unsupported dtype: {dtype}. "
                f"Supported: float32, float64, int32, int64"
            )
        
        # Create a new config object for decompression
        cdef szConfig conf = szConfig()
        conf.setDims(*shape)
        
        # Ensure compressed data is contiguous
        if not compressed.flags['C_CONTIGUOUS']:
            compressed = np.ascontiguousarray(compressed)
        
        cdef char* compressed_ptr = <char*> cnp.PyArray_DATA(compressed)
        cdef size_t compressed_size = compressed.size
        cdef size_t num_elements = conf.num_elements
        
        # Pre-allocate NumPy array for decompressed data
        cdef cnp.ndarray result = np.empty(num_elements, dtype=dtype)
        
        # Get pointer to NumPy array data
        # Pass this to SZ_decompress - it will decompress directly into our buffer!
        cdef float* float_ptr
        cdef double* double_ptr
        cdef int32_t* int32_ptr
        cdef int64_t* int64_ptr
        
        if dtype == np.float32:
            float_ptr = <float*> cnp.PyArray_DATA(result)
            c_sz.SZ_decompress[float](
                conf.conf,
                compressed_ptr,
                compressed_size,
                float_ptr
            )
        elif dtype == np.float64:
            double_ptr = <double*> cnp.PyArray_DATA(result)
            c_sz.SZ_decompress[double](
                conf.conf,
                compressed_ptr,
                compressed_size,
                double_ptr
            )
        elif dtype == np.int32:
            int32_ptr = <int32_t*> cnp.PyArray_DATA(result)
            c_sz.SZ_decompress[int32_t](
                conf.conf,
                compressed_ptr,
                compressed_size,
                int32_ptr
            )
        elif dtype == np.int64:
            int64_ptr = <int64_t*> cnp.PyArray_DATA(result)
            c_sz.SZ_decompress[int64_t](
                conf.conf,
                compressed_ptr,
                compressed_size,
                int64_ptr
            )
        
        # Data is now in our NumPy array! (zero-copy decompression)
        # Reshape to original dimensions
        return result.reshape(shape), conf
    
    @staticmethod
    def verify(cnp.ndarray src_data, cnp.ndarray dec_data) -> Tuple[float, float, float]:
        """
        Compare decompressed data with original data.
        
        Parameters
        ----------
        src_data : numpy.ndarray
            Original data before compression
        dec_data : numpy.ndarray
            Decompressed data to verify
        
        Returns
        -------
        max_diff : float
            Maximum absolute difference
        psnr : float
            Peak Signal-to-Noise Ratio in dB
        nrmse : float
            Normalized Root Mean Square Error
        """
        # Check shapes match by comparing size and ndim
        if src_data.ndim != dec_data.ndim or src_data.size != dec_data.size:
            raise ValueError("Shape mismatch between src_data and dec_data")
        
        if src_data.dtype != dec_data.dtype:
            raise ValueError("Dtype mismatch between src_data and dec_data")
        
        # Calculate data range and difference
        cdef double data_range = np.max(src_data) - np.min(src_data)
        cdef cnp.ndarray diff = src_data - dec_data
        cdef double max_diff = np.max(np.abs(diff))
        
        # Calculate MSE and derived metrics
        cdef double mse = np.mean(diff ** 2)
        cdef double nrmse = np.sqrt(mse) / data_range if data_range > 0 else 0.0
        cdef double psnr = 20 * np.log10(data_range) - 10 * np.log10(mse) if mse > 0 else float('inf')
        
        return max_diff, psnr, nrmse
