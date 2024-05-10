#ifndef LIB_DIALECT_SECRET_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_SECRET_TRANSFORMS_PASSES_TD_

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

def SecretCaptureGenericAmbientScope : Pass<"secret-capture-generic-ambient-scope"> {
  let summary = "Capture the ambient scope used in a secret.generic";
  let description = [{
    For each value used in the body of a `secret.generic` op, which is defined
    in the ambient scope outside the `generic`, add it to the argument list of
    the `generic`.
  }];
  let dependentDialects = ["mlir::heir::secret::SecretDialect"];
}

def SecretMergeAdjacentGenerics : Pass<"secret-merge-adjacent-generics"> {
  let summary = "Merge two adjacent generics into a single generic";
  let description = [{
    This pass merges two immedaitely sequential generics into a single
    generic. Useful as a sub-operation in some passes, and extracted into
    its own pass for testing purposes.
  }];
  let dependentDialects = ["mlir::heir::secret::SecretDialect"];
}

def SecretGenericAbsorbConstants : Pass<"secret-generic-absorb-constants"> {
  let summary = "Copy constants into a secret.generic body";
  let description = [{
    For each constant value used in the body of a `secret.generic` op, which is
    defined in the ambient scope outside the `generic`, add it's definition into
    the `generic` body.
  }];
  let dependentDialects = ["mlir::heir::secret::SecretDialect"];
}

def SecretExtractGenericBody : Pass<"secret-extract-generic-body"> {
  let summary = "Extract the bodies of all generic ops into functions";
  let description = [{
    This pass extracts the body of all generic ops into functions, and
    replaces the generic bodies with call ops. Used as a sub-operation in
    some passes, and extracted into its own pass for testing purposes.

    This pass works best when `--secret-generic-absorb-constants` is run
    before it so that the extracted function contains any constants used
    in the generic op's body.
  }];
  let dependentDialects = ["mlir::heir::secret::SecretDialect"];
}


#endif  // LIB_DIALECT_SECRET_TRANSFORMS_PASSES_TD_
