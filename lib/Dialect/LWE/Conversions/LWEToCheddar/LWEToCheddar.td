#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOCHEDDAR_LWETOCHEDDAR_TD_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOCHEDDAR_LWETOCHEDDAR_TD_

include "mlir/Pass/PassBase.td"

def LWEToCheddar : Pass<"lwe-to-cheddar"> {
  let summary = "Lower `lwe`/`ckks` to `cheddar` dialect.";

  let description = [{
    This pass lowers the `lwe` and `ckks` dialects to the `cheddar` dialect,
    targeting the CHEDDAR GPU FHE library.

    The pass threads a context, encoder, user interface, multiplication key
    and evaluation-key map through function arguments following the same
    pattern as the LWE-to-Lattigo pass. Rotation and conjugation keys are not
    threaded individually: the `cheddar-to-emitc` lowering reads them off the
    evaluation-key map at the use site.
  }];

  let dependentDialects = [
    "mlir::bufferization::BufferizationDialect",
    "mlir::func::FuncDialect",
    "mlir::heir::cheddar::CheddarDialect",
    "mlir::heir::ckks::CKKSDialect",
    "mlir::heir::lwe::LWEDialect",
    "mlir::heir::preprocessing::PreprocessingDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOCHEDDAR_LWETOCHEDDAR_TD_
