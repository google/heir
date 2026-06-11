# Torch CI Scripts

This directory contains scripts and models for the Torch-MLIR coverage CI job.

## CI Workflow

The CI workflow is defined in `.github/workflows/torch_mlir_coverage.yml`.

The code in this directory is intentionally not runnable within bazel because it
depends on torch-mlir development builds, and we don't want the main HEIR
project to depend on that because it is not stable (it only keeps a running
window of the last N days of nightly builds).

To run the CI workflow manually, open the workflow file above to see what
commands to run to setup a python virtualenv. Then run
`python scripts/torch_ci/entrypoint.py` from the repository base.

<!-- mdformat global-off -->
