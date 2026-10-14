#!/bin/sh
# Build the optional C/CUDA acceleration libraries for TinyLM training.
# Stdlib-only Python: the libraries are optional; ai.py falls back to pure
# Python when libtinylm_accel.so / libtinylm_cuda.so / libtinylm_rwkv.so are missing.
set -e
cd "$(dirname "$0")"

# Build C library (always)
CC=${CC:-cc}
CFLAGS=${CFLAGS:--O2 -shared -fPIC}

$CC $CFLAGS -o libtinylm_accel.so csrc/train_kernel.c -lm
echo "Built libtinylm_accel.so"

# Build CUDA library for legacy TinyLM if nvcc is available
if command -v nvcc &> /dev/null; then
    echo "CUDA compiler found, building libtinylm_cuda.so (legacy)..."
    # GTX 970 = compute capability 5.2 (Maxwell)
    ARCH=${CUDA_ARCH:-sm_52}
    nvcc -O2 -std=c++11 -arch=$ARCH -shared -Xcompiler -fPIC \
         -o csrc/libtinylm_cuda.so csrc/train_kernel_cuda.cu -lcublas -lcurand
    echo "Built csrc/libtinylm_cuda.so (arch=$ARCH)"
else
    echo "nvcc not found, skipping legacy CUDA build (install CUDA Toolkit for GPU acceleration)"
fi

# Build RWKV CUDA library if nvcc is available
if command -v nvcc &> /dev/null; then
    echo "CUDA compiler found, building libtinylm_rwkv.so (RWKV)..."
    # GTX 970 = compute capability 5.2 (Maxwell)
    ARCH=${CUDA_ARCH:-sm_52}
    nvcc -O2 -std=c++17 -arch=$ARCH -shared -Xcompiler -fPIC \
         -o csrc/libtinylm_rwkv.so csrc/rwkv_kernel.cu -lcublas -lcurand
    echo "Built csrc/libtinylm_rwkv.so (arch=$ARCH)"
else
    echo "nvcc not found, skipping RWKV CUDA build (install CUDA Toolkit for GPU acceleration)"
fi