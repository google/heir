#!/usr/bin/env bash

# Install clang and lld (container only has gcc by default)
# Pin to clang 20 — some HEIR deps don't yet support clang 21+
# (cf. https://github.com/google/heir/issues/2675).
CLANG_VERSION=20
yum install -y "clang-${CLANG_VERSION}*" "lld-${CLANG_VERSION}*"

# Install bazel
if ! bazel version; then
  arch=$(uname -m)
  if [ "$arch" == "aarch64" ]; then
    arch="arm64"
  fi
  echo "Downloading $arch Bazel binary from GitHub releases."
  bazel_version=$(cat /project/.bazelversion)
  curl -L -o $HOME/bin/bazel --create-dirs "https://github.com/bazelbuild/bazel/releases/download/${bazel_version}/bazel-${bazel_version}-linux-${arch}"
  chmod +x $HOME/bin/bazel
else
  # Bazel is installed for the correct architecture
  exit 0
fi
