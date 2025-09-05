#!/bin/bash
set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <clang_version>"
    echo "Example: $0 19"
    exit 1
fi

CLANG_VERSION=$1

# Get the current Ubuntu codename dynamically
CODENAME=$(lsb_release -cs)

# Instructions from https://apt.llvm.org/
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
echo "deb http://apt.llvm.org/${CODENAME}/ llvm-toolchain-${CODENAME}-${CLANG_VERSION} main" | sudo tee /etc/apt/sources.list.d/llvm.list
sudo apt update
sudo apt-get install -y clang-${CLANG_VERSION} lld-${CLANG_VERSION} libomp-${CLANG_VERSION}-dev

sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-${CLANG_VERSION} 200
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-${CLANG_VERSION} 200
sudo update-alternatives --install /usr/bin/lld lld /usr/bin/lld-${CLANG_VERSION} 200
