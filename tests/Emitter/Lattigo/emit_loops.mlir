// RUN: heir-translate %s --emit-lattigo | FileCheck %s

!ct = !lattigo.rlwe.ciphertext
!evaluator = !lattigo.bgv.evaluator

module attributes {scheme.bgv} {
  // CHECK: test_affine_for
  // CHECK-SAME: [[ct_init:[^ ]*]] *rlwe.Ciphertext
  // CHECK: [[iter_arg:[^ ]*]] := [[ct_init]]
  // CHECK: for [[induction_var:[^ ]*]] := 1; [[induction_var]] < 10; [[induction_var]] += 2 {
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

  // CHECK: test_scf_for
  // CHECK-SAME: [[ct_init:[^ ]*]] *rlwe.Ciphertext
  // CHECK: [[c1:[^ ]*]] := int64(1)
  // CHECK: [[c10:[^ ]*]] := int64(10)
  // CHECK: [[c2:[^ ]*]] := int64(2)
  // CHECK: [[iter_arg:[^ ]*]] := [[ct_init]]
  // CHECK: for [[induction_var:[^ ]*]] := [[c1]]; [[induction_var]] < [[c10]]; [[induction_var]] += [[c2]] {
  // CHECK:   [[op_result:[^ ,]*]], [[err0:[^ ]*]] := evaluator.RotateColumnsNew([[iter_arg]], 1)
  // CHECK:   [[iter_arg]] = [[op_result]]
  // CHECK: }
  // CHECK: [[loop_result:[^ ]*]] := [[iter_arg]]
  // CHECK: return [[loop_result]]
  func.func @test_scf_for(%evaluator: !evaluator, %ct: !ct) -> !ct {
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c2 = arith.constant 2 : index
    %1 = scf.for %arg0 = %c1 to %c10 step %c2 iter_args(%arg1 = %ct) -> (!ct) {
      %ct_12 = lattigo.bgv.rotate_columns_new %evaluator, %arg1 {static_shift = 1} : (!evaluator, !ct) -> !ct
      scf.yield %ct_12 : !ct
    }
    return %1 : !ct
  }

  // CHECK: test_affine_if
  // CHECK-SAME: [[ct_init:[^ ]*]] *rlwe.Ciphertext
  // CHECK: [[iter_arg:[^ ]*]] := [[ct_init]]
  // CHECK: for [[induction_var:[^ ]*]] := 0; [[induction_var]] < 10; [[induction_var]] += 1 {
  // CHECK:   var [[if_res:[^ ]*]] *rlwe.Ciphertext
  // CHECK:   if ([[induction_var]] % 2) == 0 {
  // CHECK:     [[op_result1:[^ ,]*]], [[err1:[^ ]*]] := evaluator.RotateColumnsNew([[iter_arg]], 1)
  // CHECK:     [[if_res]] = [[op_result1]]
  // CHECK:   } else {
  // CHECK:     [[op_result2:[^ ,]*]], [[err2:[^ ]*]] := evaluator.RotateColumnsNew([[iter_arg]], 2)
  // CHECK:     [[if_res]] = [[op_result2]]
  // CHECK:   }
  // CHECK:   [[iter_arg]] = [[if_res]]
  // CHECK: }
  // CHECK: [[loop_result:[^ ]*]] := [[iter_arg]]
  // CHECK: return [[loop_result]]
  func.func @test_affine_if(%evaluator: !evaluator, %ct: !ct) -> !ct {
    %1 = affine.for %arg0 = 0 to 10 iter_args(%arg1 = %ct) -> (!ct) {
      %2 = affine.if affine_set<(d0) : (d0 mod 2 == 0)>(%arg0) -> !ct {
        %ct_12 = lattigo.bgv.rotate_columns_new %evaluator, %arg1 {static_shift = 1} : (!evaluator, !ct) -> !ct
        affine.yield %ct_12 : !ct
      } else {
        %ct_13 = lattigo.bgv.rotate_columns_new %evaluator, %arg1 {static_shift = 2} : (!evaluator, !ct) -> !ct
        affine.yield %ct_13 : !ct
      }
      affine.yield %2 : !ct
    }
    return %1 : !ct
  }
}
