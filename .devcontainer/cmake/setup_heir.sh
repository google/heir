#!/bin/bash

# Create the build directory
mkdir -p /workspaces/heir/build && cd /workspaces/heir/build

# Configure the build
cmake -G Ninja -DMLIR_DIR=/workspaces/llvm-project/build/lib/cmake/mlir ..

# Build HEIR
cmake --build . --target all
