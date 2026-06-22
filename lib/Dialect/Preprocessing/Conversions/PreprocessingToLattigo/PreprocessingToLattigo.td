#ifndef LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOLATTIGO_PREPROCESSINGTOLATTIGO_TD_
#define LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOLATTIGO_PREPROCESSINGTOLATTIGO_TD_

include "mlir/Pass/PassBase.td"

def PreprocessingToLattigo : Pass<"preprocessing-to-lattigo", "mlir::ModuleOp"> {
  let summary = "Lower Preprocessing ops to Lattigo Plaintexts";
  let description = [{
    Converts `preprocessing.storage` to `memref` of `lattigo.rlwe.plaintext` and lowers
    `empty`, `store`, and `load` ops using linear index calculation.

    (* example filepath=tests/Dialect/Preprocessing/Conversions/doctest_lattigo.mlir *)
  }];
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::affine::AffineDialect",
    "mlir::heir::lattigo::LattigoDialect"
  ];
}

#endif  // LIB_DIALECT_PREPROCESSING_CONVERSIONS_PREPROCESSINGTOLATTIGO_PREPROCESSINGTOLATTIGO_TD_
