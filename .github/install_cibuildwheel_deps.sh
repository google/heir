#!/usr/bin/env bash

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
