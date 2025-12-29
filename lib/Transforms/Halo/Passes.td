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
  }];
  let dependentDialects = [
    "mlir::heir::secret::SecretDialect",
    "mlir::affine::AffineDialect",
    "mlir::scf::SCFDialect",
  ];
}

#endif  // LIB_TRANSFORMS_HALO_PASSES_TD_
