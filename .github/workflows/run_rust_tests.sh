#!/bin/bash

set -eux
set -o pipefail

bazel query "filter('.mlir.test$', //tests/tfhe_rust/end_to_end/...)" | xargs bazel test --sandbox_writable_path=$HOME/.cargo "$@"
