// RUN: heir-opt --lattigo-alloc-to-inplace %s | FileCheck %s

// Use the minimum level level of the two operands for the result storage

!evaluator = !lattigo.bgv.evaluator
!ct = !lattigo.rlwe.ciphertext

// CHECK: ![[evaluator:.*]] = !lattigo.bgv.evaluator

// CHECK: func.func @drop_level
// CHECK-SAME: %[[evaluator:.*]]: ![[evaluator]]
func.func @drop_level(%evaluator : !evaluator, %ct : !ct) -> !ct {
    %ct_level_0 = lattigo.bgv.rotate_columns_new %evaluator, %ct {offset = 4 : index} : (!evaluator, !ct) -> !ct
    // CHECK: %[[ct_level_2:.*]] = lattigo.rlwe.drop_level_new
    // CHECK-SAME: levelToDrop = 2
    // CHECK: %[[ct_level_4:.*]] = lattigo.rlwe.drop_level_new
    // CHECK-SAME: levelToDrop = 4
    %0 = lattigo.rlwe.drop_level_new %evaluator, %ct { levelToDrop = 2 } : (!evaluator, !ct) -> !ct
    %1 = lattigo.rlwe.drop_level_new %evaluator, %ct_level_0 { levelToDrop = 4 } : (!evaluator, !ct) -> !ct
    // CHECK: lattigo.bgv.add %[[evaluator]], %[[ct_level_2]], %[[ct_level_4]], %[[ct_level_4]]
    %2 = lattigo.bgv.add_new %evaluator, %0, %1 : (!evaluator, !ct, !ct) -> !ct
    return %2 : !ct
}
