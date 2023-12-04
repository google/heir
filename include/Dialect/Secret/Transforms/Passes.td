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
    `secret.secret` inputs. If the op has associated regions, and the operands
    are not secret, then the generic is distributed recursively through the
    op's regions as well.

    This pass is intended to be used as part of a front-end pipeline, where a
    program that operates on a secret type annotates the input to a region as
    `secret`, and then wraps the contents of the region in a single large
    `secret.generic`, then uses this pass to simplify it.

    The `distribute-through` option allows one to specify a comma-separated
    list of op names (e.g., `distribute-thorugh="affine.for,scf.if"`), which
    limits the distribution to only pass through those ops. If unset, all ops
    are distributed through when possible.
  }];
  let dependentDialects = ["mlir::heir::secret::SecretDialect"];

  let options = [
    ListOption<"opsToDistribute", "distribute-through", "std::string",
               "comma-separated list of ops that should be distributed through">,
  ];
}


#endif  // INCLUDE_DIALECT_SECRET_TRANSFORMS_PASSES_TD_
