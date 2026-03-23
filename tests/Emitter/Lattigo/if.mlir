// RUN: heir-translate %s --emit-lattigo | FileCheck %s

!ct = !lattigo.rlwe.ciphertext
!evaluator = !lattigo.bgv.evaluator

module attributes {scheme.bgv} {
  // CHECK: test_scf_if
  // CHECK-SAME: [[ct_init:[^ ]*]] *rlwe.Ciphertext
  // CHECK: [[cond:[^ ]*]] := bool(true)
  // CHECK: var [[if_result:[^ ]*]] *rlwe.Ciphertext
  // CHECK: if [[cond]] {
  // CHECK:   [[op_result:[^ ,]*]], [[err0:[^ ]*]] := evaluator.RotateColumnsNew([[ct_init]], 1)
  // CHECK:   [[if_result]] = [[op_result]]
  // CHECK: } else {
  // CHECK:   [[if_result]] = [[ct_init]]
  // CHECK: }
  // CHECK: return [[if_result]]
  func.func @test_scf_if(%evaluator: !evaluator, %ct: !ct) -> !ct {
    %0 = arith.constant 1 : i1
    %1 = scf.if %0 -> !ct {
      %ct_12 = lattigo.bgv.rotate_columns_new %evaluator, %ct {static_shift = 1} : (!evaluator, !ct) -> !ct
      scf.yield %ct_12 : !ct
    } else {
      scf.yield %ct : !ct
    }
    return %1 : !ct
  }
}
