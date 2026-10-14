#!/bin/bash
# Build CUDA acceleration library for TinyLM (GTX 970 compatible, compute capability 5.2)

set -e

echo "Building CUDA acceleration library for TinyLM..."

# Check for nvcc
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please install CUDA Toolkit."
    exit 1
fi

# Check CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
echo "CUDA version: $CUDA_VERSION"

# GTX 970 is compute capability 5.2 (Maxwell)
ARCH="sm_52"

# Build directory
BUILD_DIR="$(dirname "$0")/csrc"
mkdir -p "$BUILD_DIR"

echo "Compiling for architecture: $ARCH (GTX 970 / Maxwell)"

# Compile CUDA kernel
nvcc -O2 -std=c++11 -arch=$ARCH -shared -Xcompiler -fPIC \
     -o "$BUILD_DIR/libtinylm_cuda.so" \
     "$BUILD_DIR/train_kernel_cuda.cu" \
     -lcublas -lcurand

if [ $? -eq 0 ]; then
    echo "✓ CUDA library built successfully: $BUILD_DIR/libtinylm_cuda.so"
    
    # Test import
    python3 -c "
import sys
sys.path.insert(0, '.')
try:
    import accel_cuda
    print('✓ CUDA module loads correctly')
    print('  CUDA_AVAILABLE:', accel_cuda.CUDA_AVAILABLE)
except Exception as e:
    print('✗ CUDA module load failed:', e)
    sys.exit(1)
"
else
    echo "✗ CUDA build failed"
    exit 1
fi