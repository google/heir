#!/bin/bash

# Repository URL
REPO_URL="https://github.com/llvm/llvm-project.git"

# Extract the commit hash from bazel/import_llvm.bzl
COMMIT_HASH=$(grep -oP 'LLVM_COMMIT = "\K[0-9a-f]{40}' /workspaces/heir/bazel/import_llvm.bzl)

# Create a new directory for the MLIR repository
mkdir -p /workspaces/llvm-project && cd /workspaces/llvm-project

# This uses a somewhat round-about way to fetch only a specific commit
# thereby avoiding the (apparent) need to first shallow-clone the repo HEAD
# Note that fetching a single commit by hash is not supported by all git servers

# Initialize a new git repository
git init

# Add the target repository as a remote
git remote add origin $REPO_URL

# Fetch only the specific commit with a shallow depth
git fetch origin $COMMIT_HASH --depth 1

# Checkout the fetched commit
git checkout FETCH_HEAD

# Create a new directory for the build
mkdir -p build && cd build

# Configure the build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD=Native \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_USE_SPLIT_DWARF=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_USE_LINKER=lld \
    -DLLVM_OPTIMIZED_TABLEGEN=ON \
    -DLLVM_CCACHE_BUILD=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DMLIR_INCLUDE_INTEGRATION_TESTS=OFF

# Build MLIR
cmake --build . --target mlir-libraries
