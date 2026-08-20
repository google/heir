// RUN: heir-opt --mlir-to-ckks="greedy-level-budget=1" %s | FileCheck %s

// The current placement fails scale validation with:
//   input scale must be less than or equal to first-mod-bits - 1

// CHECK: %[[LHS:.*]] = ckks.rescale {{.*}} : {{.*}} -> !ct_L0
// CHECK: %[[BOOT_LHS:.*]] = ckks.bootstrap %[[LHS]] : !ct_L0 -> !ct_L1
// CHECK: %[[RHS:.*]] = ckks.rescale {{.*}} : {{.*}} -> !ct_L0
// CHECK: %[[BOOT_RHS:.*]] = ckks.bootstrap %[[RHS]] : !ct_L0 -> !ct_L1
// CHECK: ckks.mul %[[BOOT_LHS]], %[[BOOT_RHS]] : (!ct_L1, !ct_L1)
// CHECK-NOT: ckks.bootstrap

module attributes {backend.lattigo, scheme.ckks, backend.config_override = {bootstrapLevelsConsumed = 0 : i32}} {
  func.func @bootstrap_add_result(%x: f32 {secret.secret}) -> f32 {
    %square = arith.mulf %x, %x : f32
    %product = arith.mulf %square, %x : f32
    %out = arith.addf %product, %x : f32
    return %out : f32
  }
}
