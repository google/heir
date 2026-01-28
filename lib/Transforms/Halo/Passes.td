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

def PartialUnrollForLevelConsumption : Pass<"partial-unroll-for-level-consumption"> {
  let summary = "Partially unroll a loop over ciphertexts to better utilize level consumption.";
  let description = [{
  The `bootstrap-loop-iter-args` pass inserts a bootstrap operation at the start of each
  loop ieration, for each loop-carried iter arg. The loop body may not be sufficiently
  large to utilize all the levels provided by a bootstrap operation. This pass compensates
  for that by partially unrolling the loop so that level utilization is improved, and
  bootstraps are not required for every iteration. After unrolling, unnecessary bootstrap
  ops and `level_reduce_min` ops are removed.

  In order for a loop to qualify for this pass, it must satisfy the following properties:

  - All secret iter args have `mgmt.bootstrap` as their only use.
  - All secret values yielded in the loop body are op results of `mgmt.level_reduce_min` ops.

  This pass operates by analyzing how many levels are consumed by a loop body, and combining
  that with the maximum level in the IR to determine how much the loop can be unrolled.
  However, because this pass may be applied before the final level in the IR is chosen,
  it accepts an option `force-max-level` that allows the pass pipeline to force the loop
  unrolling calculation to use a particular value for its max level.

  (* example filepath=tests/Transforms/halo/partial_unroll_for_level_consumption_doctest.mlir *)
  }];
  let options = [
    Option<"forceMaxLevel", "force-max-level", "int", /*default=*/"0",
           "If nonzero, forces the inferred maximum level to the given value.">,
  ];
  let dependentDialects = [
    "mlir::heir::secret::SecretDialect",
    "mlir::affine::AffineDialect",
    "mlir::scf::SCFDialect",
    "mlir::heir::mgmt::MgmtDialect",
  ];
}

#endif  // LIB_TRANSFORMS_HALO_PASSES_TD_
