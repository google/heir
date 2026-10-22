# Prepare an FHE benchmarking MNIST submission

`prepare_fhe_benchmark.py` is a small, offline scaffold for the local OpenFHE
MNIST workload in [fhe-benchmarking/ml-inference][benchmark]. It combines HEIR's
generated C++ source and header with the reference stage executables, CMake
configuration, and serialization helpers. It also places the model MLIR and
trained TorchScript weights alongside them.

The scaffold still needs model-specific integration. It does not infer the
generated function's signature, weights, ciphertext layout, or cryptographic
parameters. It does not compile or validate the resulting submission. Other
workloads and remote execution are outside its scope.

## Prepare artifacts

Compile your MNIST MLIR with HEIR and export the OpenFHE source and header using
`heir-translate --emit-openfhe-pke` and
`heir-translate --emit-openfhe-pke-header`. Use
`--openfhe-include-type=embedded` to emit `#include "openfhe.h"`, matching the
reference CMake include directories. The script copies these artifacts unchanged
to `src/mlp_openfhe.cpp` and `include/mlp_openfhe.h`, so any generated includes
must use those names. If you externalize constants, copy their runtime data
files into the submission and adapt the loading paths yourself.

Supply the MLIR and TorchScript model that correspond to the generated code. All
four artifacts are required to avoid accidentally retaining the reference's old
computation, architecture, or weights.

## Create the scaffold

The following example uses the reference revision inspected for this helper:
`739378698fc1f668aff7246724b45cff76f04a98`. The reference files remain owned by
the benchmark project; HEIR neither downloads nor vendors them.

```bash
git clone https://github.com/fhe-benchmarking/ml-inference.git
cd ml-inference
git checkout 739378698fc1f668aff7246724b45cff76f04a98

# Preserve the reference; the harness expects the new submission at this path.
mv submissions/mnist submissions/mnist-reference

python /path/to/heir/scripts/prepare_fhe_benchmark.py \
  --reference submissions/mnist-reference \
  --output submissions/mnist \
  --source /path/to/generated/mlp_openfhe.cpp \
  --header /path/to/generated/mlp_openfhe.h \
  --mlir /path/to/model.mlir \
  --weights /path/to/traced_model.pt
```

The script requires Python 3.10+ and the standard library only. It can also be
run as `bazel run //scripts:prepare_fhe_benchmark -- ...` with absolute paths.
It refuses an existing output directory and checks all inputs before writing. It
copies only the listed reference sources and headers, excluding build trees and
other local files. Missing reference files produce an error rather than silently
creating an incomplete scaffold.

## Adapt and verify

Read the generated `README.md` checklist. The reference's manually written
integration has these responsibilities:

| File                                                             | What must match your model and HEIR output                                                                                                                                                                      |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/server_encrypted_compute.cpp`                               | TorchScript parameter names, shapes, argument order, and the generated function call. The reference assumes two fully connected layers and calls `mnist(cc, fc1_weight, fc1_bias, fc2_weight, fc2_bias, ctxt)`. |
| `src/mlp_encryption_utils.cpp`, `include/mlp_encryption_utils.h` | Context setup, evaluation keys, input packing, and result decoding. These depend on compilation parameters and ciphertext layouts, even when the model architecture stays the same.                             |
| `include/params.h`                                               | Dataset dimensions, instance sizes, and client/server file paths.                                                                                                                                               |
| `CMakeLists.txt`                                                 | Additional generated helpers, includes, libraries, or external constants.                                                                                                                                       |
| `docs/README.md`                                                 | Architecture changes, training, accuracy, HEIR revision, and compilation flags. The scaffold creates a TODO instead of copying old accuracy claims.                                                             |

The seven stage executables and their I/O code are reusable starting points.
Review them if the input/output contract changes. In particular, current HEIR
output can require arguments beyond the reference signature. Reuse the generated
client-interface functions when adapting encryption helpers; copying the old
packing or context settings without checking them can give incorrect results.

Keep the scaffold at `submissions/mnist`: both the server's model path and the
harness use that location. Keep the upstream copyright and license notices.
Follow the benchmark repository's dependency and build instructions, then run
from its root:

```bash
python harness/run_submission.py 0 --seed 3 --num_runs 1 --dataset mnist --model mlp
```

Inspect the correctness check and measured output before submitting. This helper
only packages files; a successful packaging run does not establish benchmark
correctness or performance.

<!-- mdformat global-off -->

[benchmark]: https://github.com/fhe-benchmarking/ml-inference/tree/739378698fc1f668aff7246724b45cff76f04a98
