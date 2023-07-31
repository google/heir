#ifndef INCLUDE_DIALECT_SECRET_TRANSFORMS_PASSES_TD_
#define INCLUDE_DIALECT_SECRET_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def SecretForgetSecrets : Pass<"secret-forget-secrets"> {
  let summary = "Convert secret types to standard types";
  let description = [{
    Drop the `secret<...>` type from the IR, replacing it with the contained
    type and the corresponding cleartext computation.
  }];
  let dependentDialects = ["mlir::heir::secret::SecretDialect"];
}

#endif  // INCLUDE_DIALECT_SECRET_TRANSFORMS_PASSES_TD_
