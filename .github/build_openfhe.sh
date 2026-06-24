#!/usr/bin/env bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
INSTALL_DIR="$WORKSPACE_DIR/openfhe-install"

# If already installed, just export variables and exit
if [ -d "$INSTALL_DIR" ] && [ -d "$INSTALL_DIR/include/openfhe" ]; then
  echo "OpenFHE already installed at $INSTALL_DIR"
  if [ -n "$GITHUB_ENV" ]; then
    echo "OPENFHE_LIB_DIR=$INSTALL_DIR/lib" >> "$GITHUB_ENV"
    INC_DIR="$INSTALL_DIR/include/openfhe"
    echo "OPENFHE_INCLUDE_DIR=$INSTALL_DIR/include:$INC_DIR:$INC_DIR/binfhe:$INC_DIR/core:$INC_DIR/pke" >> "$GITHUB_ENV"
  fi
  exit 0
fi

cd "$WORKSPACE_DIR"

# Clone OpenFHE
if [ ! -d "openfhe-development" ]; then
  git clone --depth 1 --branch v1.4.2 https://github.com/openfheorg/openfhe-development.git
fi

# Build OpenFHE
mkdir -p openfhe-development/build
cd openfhe-development/build

mkdir -p "$INSTALL_DIR"

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_UNITTESTS=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"

make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
make install

# Export variables to GITHUB_ENV if running in GitHub Actions
if [ -n "$GITHUB_ENV" ]; then
  echo "OPENFHE_LIB_DIR=$INSTALL_DIR/lib" >> "$GITHUB_ENV"
  INC_DIR="$INSTALL_DIR/include/openfhe"
  echo "OPENFHE_INCLUDE_DIR=$INSTALL_DIR/include:$INC_DIR:$INC_DIR/binfhe:$INC_DIR/core:$INC_DIR/pke" >> "$GITHUB_ENV"
fi
