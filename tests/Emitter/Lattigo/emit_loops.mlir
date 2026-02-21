// RUN: heir-translate %s --emit-lattigo | FileCheck %s

!ct = !lattigo.rlwe.ciphertext
!evaluator = !lattigo.bgv.evaluator

module attributes {scheme.bgv} {
  // CHECK: test_affine_for
  // CHECK: [[iter_arg:[^ ]*]] :=
  // CHECK: for [[induction_var:[^ ]*]] := int64(1); [[induction_var]] < 10; [[induction_var]] += 2 {
  // CHECK:   [[op_result:[^ ,]*]], [[err0:[^ ]*]] := evaluator.RotateColumnsNew([[iter_arg]], 1)
  // CHECK:   [[iter_arg]] = [[op_result]]
  // CHECK: }
  // CHECK: [[loop_result:[^ ]*]] := [[iter_arg]]
  // CHECK: return [[loop_result]]
  func.func @test_affine_for(%evaluator: !evaluator, %ct: !ct) -> !ct {
    %1 = affine.for %arg0 = 1 to 10 step 2 iter_args(%arg1 = %ct) -> (!ct) {
      %ct_12 = lattigo.bgv.rotate_columns_new %evaluator, %arg1 {static_shift = 1} : (!evaluator, !ct) -> !ct
      affine.yield %ct_12 : !ct
    }
    return %1 : !ct
  }
}
