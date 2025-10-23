---
title: Pipelines
weight: 60
---

This page documents the pass pipelines available in HEIR for the `heir-opt` and
`heir-translate` tools.

## `heir-opt`

`heir-opt` provides several pipelines to lower MLIR programs from standard
dialects to FHE dialects.

### `--heir-simd-vectorizer`

Convert FHE programs with naive loops that operate on scalar types to equivalent
programs that operate on vectors. This corresponds to the optimizations of the
[HECO compiler](https://github.com/MarbleHE/HECO).

This pass is intended to process FHE programs that are known to be good for
SIMD, but a specific SIMD-style FHE scheme (BGV, BFV, CKKS) has not yet been
chosen. It expects to handle `arith` ops operating on `tensor` types (with or
without `secret.generic`).

The pass unrolls all loops, and assumes the input data can be packed in a single
ciphertext, interpreted as vectors of slots. For well-structured loops, the
resulting SIMD operations can be converted to use minimal ciphertext rotation
ops.

### `--mlir-to-cggi`

Converts MLIR IR to the CGGI dialect defined by HEIR. It can either booleanize
the IR and optimize the circuit using Yosys optimizations, or convert integer
arithmetic to CGGI.

This pipeline has a `dataType` option, which can be `Bool` or `Integer`.

- When `dataType` is `Bool`, the pipeline first bufferizes and applies affine
  transformations. Then, it uses Yosys to synthesize the logic into a boolean or
  small-integer arithmetic circuit using `comb.truth_table` ops to represent
  programmable bootstrap operations. Finally, it converts the boolean circuit to
  the CGGI dialect.
- When `dataType` is `Integer`, the pipeline converts `arith` dialect operations
  to the CGGI dialect. This is useful for targeting backends that have native
  support for high-bitwidth FHE arithmetic, such as `tfhe-rs`.

The pass requires that the environment variable `HEIR_ABC_BINARY` contains the
location of the ABC binary and that `HEIR_YOSYS_SCRIPTS_DIR` contains the
location of the Yosys' techlib files that are needed to execute the path. This
is only needed when `dataType` is `Bool`.

### `--mlir-to-secret-arithmetic`

Converts a function using standard MLIR dialects to the `secret` dialect with
arithmetic operations. This pipeline is a precursor to lowering to specific
RLWE-based FHE schemes. It performs several transformations:

- Applies data-oblivious transforms to remove data-dependent control flow.
- Performs layout optimization to efficiently pack data in ciphertexts.
- Adds a client interface with helper functions for encryption and decryption.

This pass requires secret `func.func` inputs to be annotated with the
`{secret.secret}` attribute.

### `--mlir-to-bgv`, `--mlir-to-bfv`, `--mlir-to-ckks`

These pipelines convert a function using standard MLIR dialects to a specific
RLWE-based FHE scheme: BGV, BFV, or CKKS. They all use the
`--mlir-to-secret-arithmetic` pipeline to perform the initial lowering to secret
arithmetic. Then, they perform scheme-specific transformations, including:

- Inserting and optimizing ciphertext management operations like modulus
  switching, relinearization, and bootstrapping.
- Performing scheme-specific parameter generation and noise analysis.
- Lowering the `secret` dialect to the target FHE dialect (`bgv`, `bfv`, or
  `ckks`).

### `--scheme-to-openfhe`

Converts code expressed at the FHE scheme level (BGV, BFV, CKKS) to the
`openfhe` dialect, from which `heir-translate` can generate C++ code using the
OpenFHE library, or else the MLIR can be interpreted using OpenFHE calls.

### `--scheme-to-lattigo`

Converts code expressed at the FHE scheme level (BGV, BFV, CKKS) to the
`lattigo` dialect, from which `heir-translate` can generate Go code using the
Lattigo library.

### `--scheme-to-tfhe-rs`

Converts code expressed in the CGGI dialect to the `tfhe_rust` dialect, from
which `heir-translate` can generate Rust code using the `thfe-rs` library.

### `--scheme-to-fpt`

Converts code expressed in the CGGI dialect to the `tfhe_rust_bool` dialect.
This pipeline is used for targeting the FPT and Belfort FPGA mirrors of the
`tfhe-rs` API.

### `--scheme-to-jaxite`

Converts code expressed in the CGGI dialect to the `jaxite` dialect, from which
`heir-translate` can generate Python code using the Jaxite library.

### `--torch-linalg-to-ckks`

Converts a `linalg` MLIR program exported from PyTorch to the CKKS FHE scheme.
It first applies `linalg` preprocessing passes and then uses the
`--mlir-to-ckks` pipeline.

### `--convert-to-data-oblivious`

Transforms a program to be data-oblivious by converting control flow on secret
data (e.g., `if`, `for`, `while`) into data-independent operations.

### `--math-to-polynomial-approximation`

Approximates math operations that cannot be expressed in FHE using polynomial
approximations. This is a sub-pipeline of most other pipelines, exposed for
testing purposes.

## `heir-translate`

`heir-translate` is a tool for translating MLIR dialects to various output
formats. `heir-translate` supports the following emitters:

- `--emit-function-info`: Emits function signature information.
- `--emit-jaxite`: Emits Python code for the Jaxite TPU library's CGGI
  implementation.
- `--emit-jaxiteword`: Emits Python code for the Jaxite TPU library's CKKS
  implementation.
- `--emit-lattigo`: Emits Go code for the Lattigo library.
- `--emit-metadata`: Emits a json object describing function signatures.
- `--emit-openfhe-pke-header`: Emits a C++ header for the OpenFHE library.
- `--emit-openfhe-pke-pybind`: Emits pybind11 bindings for the OpenFHE library.
- `--emit-openfhe-pke`: Emits C++ code for the OpenFHE library.
- `--emit-simfhe`: Exports code that can be evaluated with the SimFHE simulator.
- `--emit-tfhe-rust-bool`: Emits `tfhe-rs` code against the boolean API.
- `--emit-tfhe-rust-hl`: Emits `tfhe-rs` code against the integer API.
- `--emit-tfhe-rust`: Emits `tfhe-rs` code against the `shortint` API.
- `--emit-verilog`: Emits verilog code for `arith` and `memref` programs. Used
  for integration with Yosys.

<!-- mdformat global-off -->
