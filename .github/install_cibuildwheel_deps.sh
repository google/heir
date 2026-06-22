#!/usr/bin/env bash
set -e

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
fi

manylinux-install-clang -v v20.1.8.0

# Install OpenMP development headers using the CRB repository
dnf install -y --enablerepo=crb libomp-devel

# Symlink OpenMP headers from system Clang to static Clang's include directory
for system_clang_dir in /usr/lib/clang/* /usr/lib64/clang/*; do
  [ -d "$system_clang_dir/include" ] || continue
  for f in "$system_clang_dir/include"/omp*.h; do
    [ -f "$f" ] || continue
    for static_clang_include in /opt/clang/lib/clang/*/include; do
      [ -d "$static_clang_include" ] || continue
      target="$static_clang_include/$(basename "$f")"
      if [ ! -e "$target" ]; then
        ln -sf "$f" "$target"
      fi
    done
  done
done
