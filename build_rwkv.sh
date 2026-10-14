#!/bin/bash
# Build RWKV CUDA acceleration library for TinyLM (GTX 970+ compatible)

set -e

echo "Building RWKV CUDA acceleration library for TinyLM..."

# Check for nvcc
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please install CUDA Toolkit."
    exit 1
fi

# Check CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
echo "CUDA version: $CUDA_VERSION"

# GTX 970 = compute capability 5.2 (Maxwell)
# But we'll target 5.2+ for broader compatibility
ARCH=${CUDA_ARCH:-sm_52}

# Build directory
BUILD_DIR="$(dirname "$0")/csrc"
mkdir -p "$BUILD_DIR"

echo "Compiling for architecture: $ARCH (GTX 970 / Maxwell+)"

# Compile RWKV CUDA kernel
echo "Compiling RWKV kernel..."
nvcc -O2 -std=c++17 -arch=$ARCH -shared -Xcompiler -fPIC \
     -o "$BUILD_DIR/libtinylm_rwkv.so" \
     "$BUILD_DIR/rwkv_kernel.cu" \
     -lcublas -lcurand

if [ $? -eq 0 ]; then
    echo "✓ RWKV library built successfully: $BUILD_DIR/libtinylm_rwkv.so"
    
    # Test import
    python3 -c "
import sys
sys.path.insert(0, '.')
try:
    import accel_rwkv
    print('✓ RWKV module loads correctly')
    print('  RW_KV_AVAILABLE:', accel_rwkv.RW_KV_AVAILABLE)
except Exception as e:
    print('✗ RWKV module load failed:', e)
    sys.exit(1)
"
else
    echo "✗ RWKV build failed"
    exit 1
fi