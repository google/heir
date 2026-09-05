// RUN: heir-opt --mlir-to-secret-arithmetic --secret-insert-mgmt-ckks="bootstrap-waterline=2" %s | FileCheck %s

// Regression test for the level-0 double-scale bootstrap operand bug: under
// the rescale-before-mul convention, a waterline crossing's operand is a
// mul-derived value still at double scale sitting on the LAST prime of the
// residual chain, where no prime remains to rescale it away -- Lattigo's
// bootstrap then panics ("initial Q/Scale < 0.5*Q[0]/MessageRatio").
//
// The invariant: every waterline bootstrap consumes the crossing modreduce's
// RESULT (base scale), never its double-scale operand.

// CHECK: func.func @bootstrap_base_scale
// CHECK: secret.generic(%[[arg0:.*]]: !secret.secret<tensor<1x1024xf16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>})

// The first crossing: a mul-derived value at level 1 is rescaled onto the
// bottom prime, and only the rescale RESULT is bootstrapped.
// CHECK:      %[[m1:[^ ]+]] = arith.mulf {{.*}} {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>}
// CHECK-NEXT: %[[rl1:[^ ]+]] = mgmt.relinearize %[[m1]] {mgmt.mgmt = #mgmt.mgmt<level = 1>}
// CHECK-NEXT: %[[mr0:[^ ]+]] = mgmt.modreduce %[[rl1]] {mgmt.mgmt = #mgmt.mgmt<level = 0>}
// CHECK-NEXT: %[[boot0:[^ ]+]] = mgmt.bootstrap %[[mr0]] {mgmt.mgmt = #mgmt.mgmt<level = 2>}
// CHECK-NEXT: %[[m2:[^ ]+]] = arith.mulf %[[boot0]], %[[boot0]] {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>}

// The second crossing: again the bootstrap operand is the rescale result on
// the bottom prime.
// CHECK:      %[[mr1:[^ ]+]] = mgmt.modreduce {{.*}} {mgmt.mgmt = #mgmt.mgmt<level = 0>}
// CHECK-NEXT: %[[boot1:[^ ]+]] = mgmt.bootstrap %[[mr1]] {mgmt.mgmt = #mgmt.mgmt<level = 2>}
// CHECK-NEXT: %[[m3:[^ ]+]] = arith.mulf %[[boot1]], %[[boot1]] {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>}

// No bootstrap may consume anything but a modreduce result.
// CHECK-NOT: mgmt.bootstrap

module attributes {backend.lattigo, scheme.ckks, backend.config_override = {bootstrapLevelsConsumed = 0 : i32}} {
func.func @bootstrap_base_scale(
    %x : f16 {secret.secret}
  ) -> f16 {
    %0 = arith.mulf %x, %x : f16
    %1 = arith.mulf %0, %0 : f16
    %2 = arith.mulf %1, %1 : f16
    %3 = arith.mulf %2, %2 : f16
    %4 = arith.mulf %3, %3 : f16
  return %4 : f16
}
}
