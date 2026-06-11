// RUN: heir-opt --ilp-bootstrap-placement=bootstrap-waterline=3 %s | FileCheck %s

// Orbit-style decoding can place scale management at a shared node or at a
// later node when that is cheaper. This case checks that a CKKS run decodes a
// scale-changing mgmt.modreduce only where the ILP assignment needs it.

// CHECK: func.func @orbit_output_rescale_placement
// CHECK: %[[SHARED:.*]] = arith.mulf %input0, %input1
// CHECK-NEXT: %[[RELIN:.*]] = mgmt.relinearize %[[SHARED]]
// CHECK: arith.addf
// CHECK: arith.subf
// CHECK: arith.addf
// CHECK: mgmt.adjust_scale
// CHECK-NEXT: mgmt.modreduce
// CHECK-NOT: mgmt.bootstrap

!pt_ty = tensor<8xf32>
!ct_ty = !secret.secret<!pt_ty>

module attributes {scheme.ckks} {
  func.func @orbit_output_rescale_placement(
      %arg0: !ct_ty, %arg1: !ct_ty)
      -> (!ct_ty {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 51>}) {
    %0 = secret.generic(%arg0: !ct_ty, %arg1: !ct_ty) {
    ^body(%input0: !pt_ty, %input1: !pt_ty):
      %shared = arith.mulf %input0, %input1 : !pt_ty
      %use0 = arith.addf %shared, %input0 : !pt_ty
      %use1 = arith.subf %shared, %input1 : !pt_ty
      %out = arith.addf %use0, %use1 : !pt_ty
      secret.yield %out : !pt_ty
    } -> (!ct_ty {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 51>})
    return %0 : !ct_ty
  }
}
