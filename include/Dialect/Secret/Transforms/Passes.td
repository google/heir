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

def SecretDistributeGeneric : Pass<"secret-distribute-generic"> {
  let summary = "Distribute `generic` ops through their bodies.";
  let description = [{
    Converts `generic` ops whose region contains many ops into smaller
    sequences of generic ops whose regions contain a single op, dropping the
    `generic` part from any resulting `generic` ops that have no
    `secret.secret` inputs.

    This pass is intended to be used as part of a front-end pipeline, where a
    program that operates on a secret type annotates the input to a region as
    `secret`, and then wraps the contents of the region in a single large
    `secret.generic`, then uses this pass to simplify it.
  }];
  let dependentDialects = ["mlir::heir::secret::SecretDialect"];
}


#endif  // INCLUDE_DIALECT_SECRET_TRANSFORMS_PASSES_TD_
