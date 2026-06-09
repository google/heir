#ifndef LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOMEMREF_PREPROCESSINGTOMEMREF_TD_
#define LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOMEMREF_PREPROCESSINGTOMEMREF_TD_

include "mlir/Pass/PassBase.td"

def PreprocessingToMemref : Pass<"preprocessing-to-memref", "mlir::ModuleOp"> {
  let summary = "Lower Preprocessing ops to Memref";
  let description = [{
    Converts `preprocessing.storage` to `memref` and lowers `empty`, `store`,
    and `load` ops using linear index calculation based on enclosing loops.

    (* example filepath=tests/Dialect/Preprocessing/Conversions/doctest.mlir *)
  }];
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::affine::AffineDialect"
  ];
}

#endif  // LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOMEMREF_PREPROCESSINGTOMEMREF_TD_
