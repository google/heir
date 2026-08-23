// RUN: heir-translate %s --emit-lattigo --split-input-file | FileCheck %s

// Lattigo clamps an in-place result to min(operand levels, receiver level), and
// rlwe.Element.Copy does not touch the receiver's level at all. A buffer reused
// by --lattigo-alloc-to-inplace must therefore be given the operand's level
// before the call, or the result silently inherits the buffer's (shorter)
// modulus chain and a later Rescale fails with "input Ciphertext level is too
// low".

!evaluator = !lattigo.ckks.evaluator
!ct = !lattigo.rlwe.ciphertext

module attributes {scheme.ckks} {
  // CHECK: func Binops
  // CHECK-SAME: ([[eval:[^ ]*]] *ckks.Evaluator, [[lhs:[^ ]*]] *rlwe.Ciphertext, [[rhs:[^ ]*]] *rlwe.Ciphertext, [[dst:[^ ]*]] *rlwe.Ciphertext)
  func.func @binops(%evaluator: !evaluator, %lhs: !ct, %rhs: !ct, %dst: !ct) -> !ct {
    // CHECK: [[dst]].Resize([[dst]].Degree(), [[lhs]].Level())
    // CHECK-NEXT: {{.*}} := [[eval]].Add([[lhs]], [[rhs]], [[dst]])
    %added = lattigo.ckks.add %evaluator, %lhs, %rhs, %dst : (!evaluator, !ct, !ct, !ct) -> !ct
    // CHECK: [[rhs]].Resize([[rhs]].Degree(), [[dst]].Level())
    // CHECK-NEXT: {{.*}} := [[eval]].Rescale([[dst]], [[rhs]])
    %rescaled = lattigo.ckks.rescale %evaluator, %added, %rhs : (!evaluator, !ct, !ct) -> !ct
    return %rescaled : !ct
  }

  // A receiver that is already the operand needs no reshaping.
  // CHECK: func Self_inplace
  // CHECK-NOT: Resize
  // CHECK: {{.*}}.Add(
  func.func @self_inplace(%evaluator: !evaluator, %lhs: !ct, %rhs: !ct) -> !ct {
    %added = lattigo.ckks.add %evaluator, %lhs, %rhs, %lhs : (!evaluator, !ct, !ct, !ct) -> !ct
    return %added : !ct
  }

  // The alias check covers every operand, not just the first one.
  // CHECK: func Rhs_inplace
  // CHECK-NOT: Resize
  // CHECK: {{.*}}.Add({{.*}}, {{.*}}, {{.*}})
  func.func @rhs_inplace(%evaluator: !evaluator, %lhs: !ct, %rhs: !ct) -> !ct {
    %added = lattigo.ckks.add %evaluator, %lhs, %rhs, %rhs : (!evaluator, !ct, !ct, !ct) -> !ct
    return %added : !ct
  }

  // Copy leaves the receiver's level (and its stale top limbs) in place, so a
  // negate into a different buffer has to reshape it first.
  // CHECK: func Negate_into_other_buffer
  // CHECK-SAME: ([[eval2:[^ ]*]] *ckks.Evaluator, [[in:[^ ]*]] *rlwe.Ciphertext, [[out:[^ ]*]] *rlwe.Ciphertext)
  func.func @negate_into_other_buffer(%evaluator: !evaluator, %in: !ct, %out: !ct) -> !ct {
    // CHECK: [[out]].Resize([[in]].Degree(), [[in]].Level())
    // CHECK-NEXT: [[out]].Copy([[in]])
    %negated = lattigo.rlwe.negate %evaluator, %in, %out : (!evaluator, !ct, !ct) -> !ct
    return %negated : !ct
  }
}

// -----

!evaluator = !lattigo.bgv.evaluator
!ct = !lattigo.rlwe.ciphertext

module attributes {scheme.bgv} {
  // BGV column rotation uses a dedicated emitter path, but needs the same
  // receiver-level guarantee as the generic in-place operations.
  // CHECK: func Bgv_rotate_columns
  // CHECK-SAME: ([[eval:[^ ]*]] *bgv.Evaluator, [[in:[^ ]*]] *rlwe.Ciphertext, [[out:[^ ]*]] *rlwe.Ciphertext)
  func.func @bgv_rotate_columns(%evaluator: !evaluator, %in: !ct, %out: !ct) -> !ct {
    // CHECK: [[out]].Resize([[out]].Degree(), [[in]].Level())
    // CHECK-NEXT: {{.*}} := [[eval]].RotateColumns([[in]], 1, [[out]])
    %rotated = lattigo.bgv.rotate_columns %evaluator, %in, %out {static_shift = 1} : (!evaluator, !ct, !ct) -> !ct
    return %rotated : !ct
  }
}
