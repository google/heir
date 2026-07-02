#ifndef LIB_DIALECT_ROTOM_TRANSFORMS_OUTLINEKERNELS_OUTLINEKERNELS_TD_
#define LIB_DIALECT_ROTOM_TRANSFORMS_OUTLINEKERNELS_OUTLINEKERNELS_TD_

include "mlir/Pass/PassBase.td"

def OutlineKernels
    : Pass<"rotom-outline-kernels", "ModuleOp"> {
  let summary = "Outline layout-assigned Rotom kernels into functions";
  let description = [{
    Runs after layout assignment. Each tensor operator whose Rotom lowering
    is a pure function of its layout signature (currently `linalg.matmul`
    carrying the `rotom.matmul` layout combination) is moved into a private
    function named after the op and deduplicated by signature: two ops with
    the same operand/result layouts share one callee. Call sites carry the
    result layout, so the materializer and the ciphertext lowering process
    the callee bodies like any other function.

    This keeps the Rotom kernel vocabulary contained in small per-kernel
    functions (unit-testable in isolation, deduplicated across a model) while
    the surrounding IR only sees calls. Pipelines inline the calls after the
    ciphertext lowering so cross-kernel optimization and backends without
    call support see a flat function.
  }];
  let dependentDialects = [
    "mlir::heir::rotom::RotomDialect",
    "mlir::func::FuncDialect",
  ];
}

#endif  // LIB_DIALECT_ROTOM_TRANSFORMS_OUTLINEKERNELS_OUTLINEKERNELS_TD_
