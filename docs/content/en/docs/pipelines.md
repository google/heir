---
title: Pipelines
weight: 3
---

The `--heir-tosa-to-arith` pipeline lowers a TOSA MLIR model to one that only
contains arithmetic operations via a TOSA to `linalg` lowering path. As part of
this lowering, `tensor` values are lowered to `memref`s. This introduces
`memref`s that hold intermediate computation. To simplify the model, we can
inline global constant `memref`s, expand `memref` aliasing and copy operations
and then forward values through the model using the `AffineScalarReplacement`
passes.

The pass pipeline assumes that the model is a valid TOSA MLIR model with
stripped quantized types. The
[iree-import-tflite](https://openxla.github.io/iree/getting-started/tflite/)
tool can lower a TFLite FlatBuffer to textual MLIR with
`--output-format=mlir-ir`. See
[hello_world.tosa.mlir](../tests/hello_world.tosa.mlir) for an example.
