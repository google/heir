#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOCHEDDAR_LWETOCHEDDAR_TD_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOCHEDDAR_LWETOCHEDDAR_TD_

include "mlir/Pass/PassBase.td"

def LWEToCheddar : Pass<"lwe-to-cheddar"> {
  let summary = "Lower `lwe`/`ckks` to `cheddar` dialect.";

  let description = [{
    This pass lowers the `lwe` and `ckks` dialects to the `cheddar` dialect,
    targeting the CHEDDAR GPU FHE library. It also converts `orion` dialect
    ops (linear_transform, chebyshev) to their CHEDDAR equivalents.

    The pass threads a context, encoder, and evk_map through function arguments
    following the same pattern as the LWE-to-Lattigo pass.
  }];

  let dependentDialects = [
    "mlir::heir::cheddar::CheddarDialect",
    "mlir::heir::ckks::CKKSDialect",
    "mlir::heir::lwe::LWEDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOCHEDDAR_LWETOCHEDDAR_TD_
