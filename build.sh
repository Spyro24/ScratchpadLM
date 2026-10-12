#!/bin/sh
# Build the optional C acceleration library for TinyLM training.
# Stdlib-only Python: the library is optional; ai.py falls back to pure
# Python when libtinylm_accel.so is missing.
set -e
cd "$(dirname "$0")"

CC=${CC:-cc}
CFLAGS=${CFLAGS:--O2 -shared -fPIC}

$CC $CFLAGS -o libtinylm_accel.so csrc/train_kernel.c -lm
echo "Built libtinylm_accel.so"
