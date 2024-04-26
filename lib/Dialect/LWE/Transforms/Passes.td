#ifndef LIB_DIALECT_LWE_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_LWE_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def SetDefaultParameters : Pass<"lwe-set-default-parameters"> {
  let summary = "Set default parameters for LWE ops";
  let description = [{
  This pass adds default parameters to all `lwe` types as the `lwe_params`
  attribute, and for `lwe` ops as the `params` attribute, overriding any
  existing attributes set with those names.

  This pass is primarily for testing purposes, and as a parameter provider
  before a proper parameter selection mechanism is added. This pass should not
  be used in production.

  The specific parameters are hard-coded in
  `lib/Dialect/LWE/Transforms/SetDefaultParameters.cpp`.
  }];
  let dependentDialects = ["mlir::heir::lwe::LWEDialect"];
}

#endif  // LIB_DIALECT_LWE_TRANSFORMS_PASSES_TD_
