#ifndef LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOOPENFHE_PREPROCESSINGTOOPENFHE_TD_
#define LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOOPENFHE_PREPROCESSINGTOOPENFHE_TD_

include "mlir/Pass/PassBase.td"

def PreprocessingToOpenfhe : Pass<"preprocessing-to-openfhe", "mlir::ModuleOp"> {
  let summary = "Lower Preprocessing ops to OpenFHE Plaintexts";
  let description = [{
    Converts `preprocessing.storage` to `memref` of `openfhe.plaintext` and lowers
    `empty`, `store`, and `load` ops using linear index calculation.

    (* example filepath=tests/Dialect/Preprocessing/Conversions/doctest_openfhe.mlir *)
  }];
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::affine::AffineDialect",
    "mlir::heir::openfhe::OpenfheDialect"
  ];
}

#endif  // LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOOPENFHE_PREPROCESSINGTOOPENFHE_TD_
