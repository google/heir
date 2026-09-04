#ifndef LIB_DIALECT_ROTOM_TRANSFORMS_NORMALIZECONTRACTIONS_NORMALIZECONTRACTIONS_TD_
#define LIB_DIALECT_ROTOM_TRANSFORMS_NORMALIZECONTRACTIONS_NORMALIZECONTRACTIONS_TD_

include "mlir/Pass/PassBase.td"

def NormalizeContractions
    : Pass<"rotom-normalize-contractions", "ModuleOp"> {
  let summary = "Normalize contractions to the matmul form the Rotom search covers";
  let description = [{
    Rewrites tensor operators with contractions into a normalized form for
    Rotom to perform layout search over.

    - `linalg.matvec` becomes a `linalg.matmul` against the vector expanded to
      a Kx1 matrix (with the result collapsed back). Torch-exported models
      arrive with their layers in this form after `--drop-unit-dims`.
    - A `linalg.matmul` whose accumulator init is not a zero fill -- e.g. a
      bias folded into `outs` by linalg fusion -- is split into a zero-filled
      matmul plus an explicit elementwise add of the init.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_DIALECT_ROTOM_TRANSFORMS_NORMALIZECONTRACTIONS_NORMALIZECONTRACTIONS_TD_
