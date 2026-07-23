// RUN: heir-opt --lattigo-alloc-to-inplace %s | FileCheck %s

// Use the minimum level level of the two operands for the result storage

!evaluator = !lattigo.bgv.evaluator
!ckks_evaluator = !lattigo.ckks.evaluator
!bootstrapping_evaluator = !lattigo.ckks.bootstrapping_evaluator
!ct = !lattigo.rlwe.ciphertext

// CHECK: ![[evaluator:.*]] = !lattigo.bgv.evaluator

// CHECK: func.func @drop_level
// CHECK-SAME: %[[evaluator:.*]]: ![[evaluator]], %[[ct:.*]]: ![[ct:.*]])
func.func @drop_level(%evaluator : !evaluator, %ct : !ct) -> !ct {
    %ct_level_0 = lattigo.bgv.rotate_columns_new %evaluator, %ct {static_shift = 4} : (!evaluator, !ct) -> !ct
    // CHECK: %[[ct_0:.*]] = lattigo.bgv.rotate_columns_new
    // CHECK: %[[ct_1:.*]] = lattigo.rlwe.drop_level_new %[[evaluator]], %[[ct]] {levelToDrop = 2 : i64}
    // CHECK: %[[ct_2:.*]] = lattigo.rlwe.drop_level %[[evaluator]], %[[ct_0]], %[[ct_0]] {levelToDrop = 4 : i64}
    // CHECK: %[[ct_3:.*]] = lattigo.bgv.add %[[evaluator]], %[[ct_1]], %[[ct_2]], %[[ct_0]]
    %0 = lattigo.rlwe.drop_level_new %evaluator, %ct { levelToDrop = 2 } : (!evaluator, !ct) -> !ct
    %1 = lattigo.rlwe.drop_level_new %evaluator, %ct_level_0 { levelToDrop = 4 } : (!evaluator, !ct) -> !ct
    %2 = lattigo.bgv.add_new %evaluator, %0, %1 : (!evaluator, !ct, !ct) -> !ct
    return %2 : !ct
}

// A bootstrap resets depth to zero. Do not reuse its exhausted input as
// storage for the refreshed result's users, or the user will be coereced
// down to level 0, which will then cause failures down the line.
// CHECK: func.func @bootstrap_resets_level
func.func @bootstrap_resets_level(%evaluator: !ckks_evaluator, %bootstrapping_evaluator: !bootstrapping_evaluator, %ct: !ct) -> !ct {
    // CHECK: %[[LOW:.*]] = lattigo.ckks.rescale_new
    %0 = lattigo.ckks.rescale_new %evaluator, %ct : (!ckks_evaluator, !ct) -> !ct
    // CHECK: %[[BOOT:.*]] = lattigo.ckks.bootstrap %{{.*}}, %[[LOW]]
    %1 = lattigo.ckks.bootstrap %bootstrapping_evaluator, %0 : (!bootstrapping_evaluator, !ct) -> !ct
    // CHECK: %[[MUL:.*]] = lattigo.ckks.mul %{{.*}}, %[[BOOT]], %[[BOOT]], %[[BOOT]]
    %2 = lattigo.ckks.mul_new %evaluator, %1, %1 : (!ckks_evaluator, !ct, !ct) -> !ct
    // CHECK: %[[RESCALED:.*]] = lattigo.ckks.rescale %{{.*}}, %[[MUL]], %[[LOW]]
    %3 = lattigo.ckks.rescale_new %evaluator, %2 : (!ckks_evaluator, !ct) -> !ct
    return %3 : !ct
}
