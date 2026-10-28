"""Package HEIR OpenFHE artifacts with the ML-inference MNIST reference."""

import argparse
from pathlib import Path
import shutil


# Reuse the reference's stage executables and I/O contract without vendoring it.
# Deliberately exclude its generated computation, trained weights, and build tree.
REFERENCE_FILES = (
    "CMakeLists.txt",
    "include/mlp_encryption_utils.h",
    "include/params.h",
    "include/utils.h",
    "src/client_key_generation.cpp",
    "src/client_preprocess_input.cpp",
    "src/client_encode_encrypt_input.cpp",
    "src/client_decrypt_decode.cpp",
    "src/client_postprocess.cpp",
    "src/server_preprocess_model.cpp",
    "src/server_encrypted_compute.cpp",
    "src/mlp_encryption_utils.cpp",
)

ADAPTER_CHECKLIST = """# HEIR MNIST submission scaffold

This directory reuses the fhe-benchmarking/ml-inference MNIST reference.
Packaging does not establish compatibility with the supplied HEIR artifacts.
Review and adapt these files before building or submitting:

- `src/server_encrypted_compute.cpp`: model parameter names, shapes, order,
  and the call to the generated function (including any additional arguments).
- `src/mlp_encryption_utils.cpp` and `include/mlp_encryption_utils.h`:
  context parameters, evaluation keys, input packing, and output decoding.
  Match these to the generated client-interface functions and ciphertext layout.
- `include/params.h`: input/output dimensions and the benchmark I/O contract.
- `CMakeLists.txt`: any additional sources, headers, or runtime dependencies.
- `docs/README.md`: describe the architecture, training, accuracy, compiler
  revision, compilation flags, and any changes to the benchmark model.

The supplied artifacts were copied to `src/mlp_openfhe.cpp`,
`include/mlp_openfhe.h`, `docs/mlp.mlir`, and `data/traced_model.pt`.
Their contents were not rewritten. Any includes must match the packaged names.
Supply additional files manually if the generated code uses external constants.

Place this directory at `submissions/mnist` in the benchmark checkout: the
reference server and harness use that path. Build and run the benchmark from
the checkout root after adapting the files above. Keep upstream license notices.

<!-- mdformat global-off -->
"""


def prepare_submission(
    reference: Path,
    output: Path,
    *,
    source: Path,
    header: Path,
    mlir: Path,
    weights: Path,
) -> None:
  """Create a new scaffold; leave the reference and supplied artifacts intact."""
  reference = reference.resolve()
  output = output.resolve()
  if output.exists():
    raise FileExistsError(f"Output already exists: {output}")
  if output.is_relative_to(reference):
    raise ValueError("Output must be outside the reference directory")

  files = {name: reference / name for name in REFERENCE_FILES}
  files.update({
      "src/mlp_openfhe.cpp": source,
      "include/mlp_openfhe.h": header,
      "docs/mlp.mlir": mlir,
      "data/traced_model.pt": weights,
  })
  # Validate everything before creating output so missing inputs leave no scaffold.
  for path in files.values():
    if not path.is_file():
      raise FileNotFoundError(f"Required file not found: {path}")

  output.mkdir(parents=True)
  for name, path in files.items():
    destination = output / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)
  (output / "README.md").write_text(ADAPTER_CHECKLIST, encoding="utf-8")
  (output / "docs/README.md").write_text(
      "# Model documentation\n\nTODO: Document the model architecture,"
      " training, accuracy, HEIR revision,\ncompilation flags, and changes to"
      " the benchmark model before submission.\n\n<!-- mdformat global-off"
      " -->\n",
      encoding="utf-8",
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--reference",
      required=True,
      type=Path,
      help="MNIST reference directory from fhe-benchmarking/ml-inference",
  )
  parser.add_argument(
      "--output", required=True, type=Path, help="New submission directory"
  )
  parser.add_argument(
      "--source", required=True, type=Path, help="HEIR C++ output"
  )
  parser.add_argument(
      "--header", required=True, type=Path, help="HEIR C++ header"
  )
  parser.add_argument(
      "--mlir", required=True, type=Path, help="Model input MLIR"
  )
  parser.add_argument(
      "--weights", required=True, type=Path, help="Trained TorchScript model"
  )
  args = parser.parse_args()
  try:
    prepare_submission(**vars(args))
  except (OSError, ValueError) as error:
    parser.error(str(error))
  print(
      f"Created {args.output}. Review README.md and adapt the model"
      " integration."
  )


if __name__ == "__main__":
  main()
