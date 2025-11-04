
// RUN: heir-opt --lattigo-hoist-rotations %s | FileCheck %s

!evaluator = !lattigo.bgv.evaluator
!ct = !lattigo.rlwe.ciphertext

module {
  func.func @simple_sum(%evaluator: !evaluator, %ct: !ct) -> !ct {
    // CHECK: lattigo.rlwe.rotate_hoisted_new
    // CHECK-COUNT-4: lattigo.rlwe.lookup_rotated
    // CHECK-NOT: lattigo.bgv.rotate_columns_new
    %ct_0 = lattigo.bgv.rotate_columns_new %evaluator, %ct {offset = 16} : (!evaluator, !ct) -> !ct
    %ct_1 = lattigo.bgv.add_new %evaluator, %ct, %ct_0 : (!evaluator, !ct, !ct) -> !ct
    %ct_2 = lattigo.bgv.rotate_columns_new %evaluator, %ct {offset = 8} : (!evaluator, !ct) -> !ct
    %ct_3 = lattigo.bgv.add_new %evaluator, %ct_1, %ct_2 : (!evaluator, !ct, !ct) -> !ct
    %ct_4 = lattigo.bgv.rotate_columns_new %evaluator, %ct {offset = 5} : (!evaluator, !ct) -> !ct
    %ct_5 = lattigo.bgv.add_new %evaluator, %ct_3, %ct_4 : (!evaluator, !ct, !ct) -> !ct
    %ct_6 = lattigo.bgv.rotate_columns_new %evaluator, %ct {offset = 12} : (!evaluator, !ct) -> !ct
    %ct_7 = lattigo.bgv.add_new %evaluator, %ct_5, %ct_6 : (!evaluator, !ct, !ct) -> !ct
    return %ct_7 : !ct
  }
}
