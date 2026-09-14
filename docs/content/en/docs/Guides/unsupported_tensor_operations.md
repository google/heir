---
title: Encrypted tensor restrictions
---

Support depends on the operations, tensor shapes, encrypted operands, and
selected compiler pipeline. A model family such as a CNN is not a guarantee that
every configuration can be compiled.

The tensor-to-ciphertext pipeline checks the following restrictions before
layout assignment. These checks also run in `--torch-linalg-to-ckks`.

| Construct on encrypted data | Current requirement                                                  | Possible adjustment                             |
| --------------------------- | -------------------------------------------------------------------- | ----------------------------------------------- |
| Tensor element extraction   | Compile-time constant indices after normalization and loop unrolling | Use a constant index when the model permits it  |
| Tensor padding              | Static input/output shapes and constant padding amounts              | Export fixed dimensions and padding amounts     |
| Padding value               | Constant zero                                                        | Use zero padding when appropriate for the model |

Cleartext tensor operations are not subject to these checks. An index expression
that folds to a constant is supported by the check; a runtime cleartext index
into an encrypted tensor is still a runtime index. These checks do not establish
that all later lowering or backend requirements are satisfied. In particular,
padding must also fit the ciphertext representation selected later in the
pipeline.

The compiler reports the offending source location when the frontend preserves
it, along with an explanation and a relevant requirement. Preserve source
locations when handing MLIR between tools:

```python
mlir_text = module.operation.get_asm(enable_debug_info=True)
```

Use `--mlir-print-debuginfo` on each `heir-opt` invocation whose output will be
parsed by another compiler step.

For custom pipelines, `--validate-tensor-kernels` runs just these checks. Place
it after normalization and static loop unrolling, before layout assignment.
Running it earlier can reject index expressions or loops that normalization
would resolve. A generic conversion failure can still indicate another
unsupported construct or a compiler defect; it is not automatically classified
as a model limitation.

<!-- mdformat global-off -->
