#!/bin/bash
# This script is a wrapper for heir-opt-bin that sets up the environment
# for standalone execution.

set -e

# When run via `bazel run`, $0 is the path to this script. The runfiles
# directory is created by bazel as a sibling directory with a .runfiles suffix.
RUNFILES_DIR="${0}.runfiles"

if [ ! -d "${RUNFILES_DIR}" ]; then
  # We may be operating inside of a test context, in which case the runfiles
  # is in a higher directory. E.g., add_one.mlir.test.runfiles might be in
  # bazel-out/k8-dbg/bin/tests/Transforms/layout_propagation/add_one.mlir.test.runfiles/_main/tools/heir-opt
  # And the runfiles dir we want is
  # bazel-out/k8-dbg/bin/tests/Transforms/layout_propagation/add_one.mlir.test.runfiles

  # Check if the current directory contains `.runfiles/` and if so, get the
  # path up to the runfiles part.
  CURRENT_DIR="$(dirname "$0")"
  if [[ "${CURRENT_DIR}" == *".runfiles"* ]]; then
    RUNFILES_DIR="${CURRENT_DIR%%.runfiles*}.runfiles"
  fi
fi

# If there's a runfiles dir, then we can set the relevant env variables
if [ -d "${RUNFILES_DIR}" ]; then
  # The main workspace is symlinked as '_main' inside the runfiles dir.
  HEIR_WORKSPACE="_main"
  ABC_WORKSPACE="edu_berkeley_abc"

  HEIR_OPT_BIN="${RUNFILES_DIR}/${HEIR_WORKSPACE}/tools/heir-opt-bin"

  # These paths correspond to the `data` dependencies of the `heir-opt-bin` target.
  export HEIR_ABC_BINARY="${RUNFILES_DIR}/${ABC_WORKSPACE}/abc"
  export HEIR_YOSYS_SCRIPTS_DIR="${RUNFILES_DIR}/${HEIR_WORKSPACE}/lib/Transforms/YosysOptimizer/yosys"
fi

# If there's no runfiles dir, we currently rely on the user to have set the
# relevant env vars. We could also attempt to auto-detect the ABC path, but not
# the yosys scripts path. If they're in the same directory as this binary, then
# this should work.

# Use exec to replace this script's process with the actual binary.
exec "${HEIR_OPT_BIN}" "$@"
