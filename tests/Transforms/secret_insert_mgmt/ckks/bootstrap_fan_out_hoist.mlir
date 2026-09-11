// RUN: heir-opt --secret-insert-mgmt-ckks="bootstrap-waterline=1 level-budget=2" %s | FileCheck %s

// A deep value fanning out to three sibling rotate+modreduce consumers, each
// of which crosses the waterline, must be refreshed ONCE before the fan-out
// (rotation preserves level, so one bootstrap of the shared operand serves
// every sibling) instead of once per sibling. The shared refresh consumes a
// pre-bootstrap rescale of the root (adjust_scale + modreduce), so the
// bootstrap operand is at base scale on a reserved prime below the root.

// CHECK: func.func @rotation_fan_out
// CHECK:      %[[ra:[^ ]+]] = mgmt.modreduce
// CHECK:      %[[deep:[^ ]+]] = arith.addf %[[ra]], %[[ra]]
// CHECK-NEXT: %[[adj0:[^ ]+]] = mgmt.adjust_scale %[[deep]]
// CHECK-NEXT: %[[rboot:[^ ]+]] = mgmt.modreduce %[[adj0]] {mgmt.mgmt = #mgmt.mgmt<level = 0>}
// CHECK-NEXT: %[[boot:[^ ]+]] = mgmt.bootstrap %[[rboot]] {mgmt.mgmt = #mgmt.mgmt<level = 2>}
// CHECK-NEXT: %[[refreshed:[^ ]+]] = arith.addf %[[boot]], %[[boot]]
// CHECK-NEXT: %[[rot1:[^ ]+]] = tensor_ext.rotate %[[refreshed]]
// CHECK-NEXT: %[[m1:[^ ]+]] = mgmt.modreduce %[[rot1]]
// CHECK-NEXT: %[[rot2:[^ ]+]] = tensor_ext.rotate %[[refreshed]]
// CHECK-NEXT: %[[m2:[^ ]+]] = mgmt.modreduce %[[rot2]]
// CHECK-NEXT: %[[rot3:[^ ]+]] = tensor_ext.rotate %[[refreshed]]
// CHECK-NEXT: %[[m3:[^ ]+]] = mgmt.modreduce %[[rot3]]
// CHECK-NOT:  mgmt.bootstrap

module attributes {backend.lattigo, scheme.ckks, backend.config_override = {bootstrapLevelsConsumed = 0 : i32}} {
func.func @rotation_fan_out(%arg0: !secret.secret<tensor<16xf32>>) -> !secret.secret<tensor<16xf32>> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %0 = secret.generic(%arg0: !secret.secret<tensor<16xf32>>) {
  ^body(%input0: tensor<16xf32>):
    %1 = arith.addf %input0, %input0 : tensor<16xf32>
    %r0 = mgmt.modreduce %1 : tensor<16xf32>
    %2 = arith.addf %r0, %r0 : tensor<16xf32>
    %r1 = mgmt.modreduce %2 : tensor<16xf32>
    // %deep sits exactly at the waterline; each sibling's own modreduce
    // would cross it.
    %deep = arith.addf %r1, %r1 : tensor<16xf32>
    %rot1 = tensor_ext.rotate %deep, %c1 : tensor<16xf32>, index
    %m1 = mgmt.modreduce %rot1 : tensor<16xf32>
    %rot2 = tensor_ext.rotate %deep, %c2 : tensor<16xf32>, index
    %m2 = mgmt.modreduce %rot2 : tensor<16xf32>
    %rot3 = tensor_ext.rotate %deep, %c3 : tensor<16xf32>, index
    %m3 = mgmt.modreduce %rot3 : tensor<16xf32>
    %s1 = arith.addf %m1, %m2 : tensor<16xf32>
    %s2 = arith.addf %s1, %m3 : tensor<16xf32>
    secret.yield %s2 : tensor<16xf32>
  } -> !secret.secret<tensor<16xf32>>
  return %0 : !secret.secret<tensor<16xf32>>
}

// A BSGS-shaped fan-out: one shared rotation consumed by two sibling
// crossing modreduces, plus a direct crossing on the deep value itself.
// The shared rotation may be looked through (all of its consumers are
// planned crossings), so the whole group still needs only ONE bootstrap.

// CHECK: func.func @shared_rotation_fan_out
// CHECK:      %[[rc:[^ ]+]] = mgmt.modreduce
// CHECK:      %[[deep2:[^ ]+]] = arith.addf %[[rc]], %[[rc]]
// CHECK-NEXT: %[[adj4:[^ ]+]] = mgmt.adjust_scale %[[deep2]]
// CHECK-NEXT: %[[rboot2:[^ ]+]] = mgmt.modreduce %[[adj4]] {mgmt.mgmt = #mgmt.mgmt<level = 0>}
// CHECK-NEXT: %[[boot2:[^ ]+]] = mgmt.bootstrap %[[rboot2]] {mgmt.mgmt = #mgmt.mgmt<level = 2>}
// CHECK-NEXT: %[[refreshed2:[^ ]+]] = arith.addf %[[boot2]], %[[boot2]]
// CHECK-NEXT: %[[rot:[^ ]+]] = tensor_ext.rotate %[[refreshed2]]
// CHECK-NOT:  mgmt.bootstrap

func.func @shared_rotation_fan_out(%arg0: !secret.secret<tensor<16xf32>>) -> !secret.secret<tensor<16xf32>> {
  %c1 = arith.constant 1 : index
  %0 = secret.generic(%arg0: !secret.secret<tensor<16xf32>>) {
  ^body(%input0: tensor<16xf32>):
    %1 = arith.addf %input0, %input0 : tensor<16xf32>
    %r0 = mgmt.modreduce %1 : tensor<16xf32>
    %2 = arith.addf %r0, %r0 : tensor<16xf32>
    %r1 = mgmt.modreduce %2 : tensor<16xf32>
    %deep = arith.addf %r1, %r1 : tensor<16xf32>
    %rot1 = tensor_ext.rotate %deep, %c1 : tensor<16xf32>, index
    %m1 = mgmt.modreduce %rot1 : tensor<16xf32>
    %m2 = mgmt.modreduce %rot1 : tensor<16xf32>
    %m3 = mgmt.modreduce %deep : tensor<16xf32>
    %s1 = arith.addf %m1, %m2 : tensor<16xf32>
    %s2 = arith.addf %s1, %m3 : tensor<16xf32>
    secret.yield %s2 : tensor<16xf32>
  } -> !secret.secret<tensor<16xf32>>
  return %0 : !secret.secret<tensor<16xf32>>
}
}
