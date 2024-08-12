#ifndef LIB_DIALECT_LWE_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_LWE_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"


def AddClientInterface : Pass<"lwe-add-client-interface"> {
  let summary = "Add client interfaces to (R)LWE encrypted functions";
  let description = [{
  This pass adds encrypt and decrypt functions for each compiled function in the
  IR. These functions maintain the same interface as the original function,
  while the compiled function may lose some of this information by the lowerings
  to ciphertext types (e.g., a scalar ciphertext, when lowered through RLWE schemes,
  must be encoded as a tensor).
  }];
  let dependentDialects = ["mlir::heir::lwe::LWEDialect"];
  let options = [
    Option<"usePublicKey", "use-public-key", "bool", /*default=*/"false",
           "If true, generate a client interface that uses a public key for encryption.">,
    Option<"oneValuePerHelperFn", "one-value-per-helper-fn", "bool", /*default=*/"false",
           "If true, split encryption helpers into separate functions for each SSA value.">
  ];
}

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
