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
}

def AddDebugPort : Pass<"lwe-add-debug-port"> {
  let summary = "Add debug port to (R)LWE encrypted functions";
  let description = [{
  This pass adds debug ports to the specified function in the IR. The debug ports
  are prefixed with "__heir_debug" and are invoked after each homomorphic operation in the
  function. The debug ports are declarations and user should provide functions with
  the same name in their code.

  For example, if the function is called "foo", the secret key is added to its
  arguments, and the debug port is called after each homomorphic operation:
  ```mlir
  // declaration of external debug function
  func.func private @__heir_debug(%sk : !sk, %ct : !ct)

  // secret key added as function arg
  func.func @foo(%sk : !sk, ...) {
    %ct = lwe.radd ...
    // invoke external debug function
    __heir_debug(%sk, %ct)
    %ct1 = lwe.rmul ...
    __heir_debug(%sk, %ct1)
    ...
  }
  ```
  }];
  let dependentDialects = ["mlir::heir::lwe::LWEDialect"];
  let options = [
    Option<"entryFunction", "entry-function", "std::string",
           /*default=*/"", "Default entry function "
           "name of entry function.">,
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
