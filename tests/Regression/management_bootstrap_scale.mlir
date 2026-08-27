// RUN: heir-opt --mlir-to-ckks="greedy-level-budget=1" %s | FileCheck %s --implicit-check-not=ckks.bootstrap --implicit-check-not=mgmt.mul_headroom

// CHECK: %[[SQUARE_L0:.*]] = ckks.rescale {{.*}} : {{.*}} -> !ct_L0
// CHECK: %[[SQUARE_L1:.*]] = ckks.bootstrap %[[SQUARE_L0]] : !ct_L0 -> !ct_L1
// CHECK: ckks.mul {{.*}}%[[SQUARE_L1]]{{.*}} : (!ct_L1, !ct_L1)

module attributes {backend.lattigo, scheme.ckks, backend.config_override = {bootstrapLevelsConsumed = 0 : i32}} {
  func.func @bootstrap_add_result(%x: f32 {secret.secret}) -> f32 {
    %square = arith.mulf %x, %x : f32
    %product = arith.mulf %square, %x : f32
    %out = arith.addf %product, %x : f32
    return %out : f32
  }
}
