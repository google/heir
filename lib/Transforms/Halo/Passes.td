#ifndef LIB_TRANSFORMS_HALO_PASSES_TD_
#define LIB_TRANSFORMS_HALO_PASSES_TD_

include "mlir/Pass/PassBase.td"

def ReconcileMixedSecretnessIterArgs : Pass<"reconcile-mixed-secretness-iter-args"> {
  let summary = "Peel the first iteration of loops with mixed secretness iter args";
  let description = [{
  Loops that involve secret iter args may have iter args whose initial values are
  plaintexts. For example, a loop that sums a set of ciphertexts may start from
  a plaintext initial value, not necessarily zero.

  In this situation, lowering the loop to ciphertext types would fail because of
  a ciphertext/plaintext type mismatch. This pass avoids this issue by peeling
  the first iteration of any such loops.

  (* example filepath=tests/Transforms/halo/reconcile_mixed_secretness_iter_args_doctest.mlir *)
  }];
  let dependentDialects = [
    "mlir::heir::secret::SecretDialect",
    "mlir::affine::AffineDialect",
    "mlir::scf::SCFDialect",
  ];
}

def BootstrapLoopIterArgs : Pass<"bootstrap-loop-iter-args"> {
  let summary = "Bootstrap loop-carried iter args at the start of each loop iteration";
  let description = [{
  Loops that involve secret iter-args have to have invariant ciphertext management
  properties across iterations of the loop. To enforce this, this pass inserts
  bootstrap ops of all loop-carried variables at the start of each loop iteration,
  while also ensuring:

  - All loop initializers are mod-switched to the lowest level before entering the loop
  - All values yielded to the next iteration of the loop are mod-switched to the lowest
    level before yielding.

  This pass is intended to preface further optimizations of a loop, by bringing
  a loop to a consistent state where the ciphertext level is invariant across
  iterations of the loop. In particular, this requires the loop has already been
  processed by a pass (such as `reconcile-mixed-secretness-iter-args`) that ensures
  iter args and initializers have equal types.

  (* example filepath=tests/Transforms/halo/bootstrap_loop_iter_args_doctest.mlir *)
  }];
  let dependentDialects = [
    "mlir::heir::secret::SecretDialect",
    "mlir::affine::AffineDialect",
    "mlir::scf::SCFDialect",
    "mlir::heir::mgmt::MgmtDialect",
  ];
}


#endif  // LIB_TRANSFORMS_HALO_PASSES_TD_
