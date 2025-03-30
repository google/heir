// RUN: heir-opt --lattigo-alloc-to-inplace %s | FileCheck %s

!evaluator = !lattigo.bgv.evaluator
!ct = !lattigo.rlwe.ciphertext

// CHECK-LABEL: func.func @drop_level
func.func @drop_level(%evaluator : !evaluator, %ct : !ct) -> !ct {
    // CHECK: lattigo.rlwe.drop_level
    // CHECK-NOT: lattigo.rlwe.drop_level_new
    %0 = lattigo.rlwe.drop_level_new %evaluator, %ct : (!evaluator, !ct) -> !ct
    return %0 : !ct
}
