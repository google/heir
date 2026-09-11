// RUN: heir-opt %s --populate-scale-ckks | FileCheck %s

// A join whose operands BOTH arrive through adjust_scale + modreduce
// chains blocks forward scale propagation on both sides; with no
// downstream anchor before the yield, the backward pass has no seed
// either, so both adjust_scale results are underdetermined. The pass
// must resolve them to the canonical target (the scale that makes the
// paired modreduce output the default scale) instead of dropping them,
// which used to leave a scale-90 operand meeting a scale-45 operand.

module attributes {backend.openfhe, ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797005856769, 35184478519297, 35184269590529, 35184474587137, 35184270114817, 35184465412097], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  // CHECK: func.func @underdetermined_both_sides
  func.func @underdetermined_both_sides(%arg0: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 5>}, %arg1: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 5>}) -> !secret.secret<f32> {
    %1 = secret.generic(%arg0 : !secret.secret<f32>, %arg1 : !secret.secret<f32>) {
    ^body(%x: f32, %y: f32):
      // Unrescaled product (scale 90) heading to the join.
      // CHECK: arith.mulf
      // CHECK-SAME: scale = 90
      %m = arith.mulf %x, %y {mgmt.mgmt = #mgmt.mgmt<level = 5, dimension = 3>} : f32
      %r = mgmt.relinearize %m {mgmt.mgmt = #mgmt.mgmt<level = 5>} : f32
      %lr = mgmt.level_reduce %r {levelToDrop = 2 : i64, mgmt.mgmt = #mgmt.mgmt<level = 3>} : f32
      // The 90-side adjust_scale resolves to 90 (identity) and is folded
      // away; its modreduce rescales to the default scale.
      // CHECK: mgmt.level_reduce
      // CHECK-SAME: scale = 90
      // CHECK-NEXT: mgmt.modreduce
      // CHECK-SAME: scale = 45
      %adj = mgmt.adjust_scale %lr {id = 0 : i64, mgmt.mgmt = #mgmt.mgmt<level = 3>} : f32
      %mr = mgmt.modreduce %adj {mgmt.mgmt = #mgmt.mgmt<level = 2>} : f32
      // The rescaled side (scale 45) resolves to the same target via a
      // materialized mul-by-ones at scale 45, then rescales back to 45.
      // CHECK: mgmt.level_reduce
      // CHECK-SAME: scale = 45
      // CHECK-NEXT: arith.mulf
      // CHECK-SAME: scale = 90
      // CHECK-NEXT: mgmt.modreduce
      // CHECK-SAME: scale = 45
      %lr2 = mgmt.level_reduce %y {levelToDrop = 2 : i64, mgmt.mgmt = #mgmt.mgmt<level = 3>} : f32
      %adj2 = mgmt.adjust_scale %lr2 {id = 1 : i64, mgmt.mgmt = #mgmt.mgmt<level = 3>} : f32
      %mr2 = mgmt.modreduce %adj2 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : f32
      // Both operands land at the default scale.
      // CHECK: arith.subf
      // CHECK-SAME: scale = 45
      %s = arith.subf %mr, %mr2 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : f32
      secret.yield %s : f32
    } -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 2>})
    return %1 : !secret.secret<f32>
  }
}
