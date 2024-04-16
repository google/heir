#ifndef INCLUDE_DIALECT_BGV_TRANSFORMS_PASSES_TD_
#define INCLUDE_DIALECT_BGV_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def AddClientInterface : Pass<"bgv-add-client-interface"> {
  let summary = "Add client interfaces to BGV encrypted functions";
  let description = [{
  This pass adds encrypt and decrypt functions for each compiled function in the
  IR. These functions maintain the same interface as the original function,
  while the compiled function may lose some of this information by the lowerings
  to ciphertext types (e.g., a scalar ciphertext, when lowered through BGV, must
  be encoded as a tensor).
  }];
  let dependentDialects = ["mlir::heir::bgv::BGVDialect"];
}

#endif  // INCLUDE_DIALECT_BGV_TRANSFORMS_PASSES_TD_
