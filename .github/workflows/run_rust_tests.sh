#!/bin/bash

set -eux
set -o pipefail

# noincompatible_strict_action_env is used to inherit the PATH environment
# variable from the parent process, which is needed to find the `cargo`
# executable.

bazel query "filter('.mlir.test$', //tests/Examples/tfhe_rust/...)" \
| xargs bazel test \
--noincompatible_strict_action_env \
-c fastbuild \
--sandbox_writable_path=$HOME/.cargo \
"$@"

bazel query "filter('.mlir.test$', //tests/Examples/tfhe_rust_hl/cpu/...)" \
| xargs bazel test \
--noincompatible_strict_action_env \
--test_timeout=360 \
-c fastbuild \
--sandbox_writable_path=$HOME/.cargo \
"$@"
