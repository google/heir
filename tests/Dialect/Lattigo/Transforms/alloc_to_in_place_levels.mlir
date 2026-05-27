// RUN: heir-opt --lattigo-alloc-to-inplace %s | FileCheck %s

// Use the minimum level level of the two operands for the result storage

!evaluator = !lattigo.bgv.evaluator
!ct = !lattigo.rlwe.ciphertext

// CHECK: ![[evaluator:.*]] = !lattigo.bgv.evaluator

// CHECK: func.func @drop_level
// CHECK-SAME: %[[evaluator:.*]]: ![[evaluator]], %[[ct:.*]]: ![[ct:.*]])
func.func @drop_level(%evaluator : !evaluator, %ct : !ct) -> !ct {
    %ct_level_0 = lattigo.bgv.rotate_columns_new %evaluator, %ct {static_shift = 4} : (!evaluator, !ct) -> !ct
    // CHECK: %[[ct_0:.*]] = lattigo.bgv.rotate_columns_new
    // CHECK: %[[ct_1:.*]] = lattigo.rlwe.drop_level %[[evaluator]], %[[ct]], %[[ct]] {levelToDrop = 2 : i64}
    // CHECK: %[[ct_2:.*]] = lattigo.rlwe.drop_level %[[evaluator]], %[[ct_0]], %[[ct_0]] {levelToDrop = 4 : i64}
    // CHECK: %[[ct_3:.*]] = lattigo.bgv.add %[[evaluator]], %[[ct_1]], %[[ct_2]], %[[ct]]
    %0 = lattigo.rlwe.drop_level_new %evaluator, %ct { levelToDrop = 2 } : (!evaluator, !ct) -> !ct
    %1 = lattigo.rlwe.drop_level_new %evaluator, %ct_level_0 { levelToDrop = 4 } : (!evaluator, !ct) -> !ct
    %2 = lattigo.bgv.add_new %evaluator, %0, %1 : (!evaluator, !ct, !ct) -> !ct
    return %2 : !ct
}
