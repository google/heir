# Unsupported / Blocked OpenFHE EmitC Tests

The following tests from the original `Openfhe` emitter test suite are currently not supported or blocked in the `OpenfheEmitC` translation pipeline due to upstream MLIR limitations or pending feature migrations:

| Test Name | Reason for Exclusion | Upstream / Component Blocked |
| :--- | :--- | :--- |
| `emit_pybind.mlir` | Pybind11 binding generation requires custom AST emission tools not yet ported to `EmitC`. | `OpenfheEmitC` |
| `emit_parallel.mlir` | OpenMP / multi-threading sections do not have an equivalent representation in standard `EmitC`. | MLIR `EmitC` Dialect |
| `emit_scf_loops.mlir` | Global variable retrieval (`emitc.get_global`) for dense tensor constants in nested modules fails module-level verification. | MLIR `ConvertToEmitC` |
| `emit_openfhe_pke.mlir` | Full PKE end-to-end pipeline relies on complex pointer structures and parameter generation. | `OpenfheEmitC` |

These tests will be incrementally onboarded as the underlying `EmitC` dialect extensions mature.
