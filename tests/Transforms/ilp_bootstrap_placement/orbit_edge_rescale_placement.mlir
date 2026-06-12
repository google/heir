// RUN: heir-opt --ilp-bootstrap-placement=bootstrap-waterline=3 %s | FileCheck %s --check-prefix=CHECK-CKKS
// RUN: heir-opt --ilp-bootstrap-placement="bootstrap-waterline=3 scale-waterline=51 scale-factor-bits=51" %s | FileCheck %s --check-prefix=CHECK-CKKS-ALIGNED

// CKKS mode is inferred from the module attribute. It keeps Sw and Sf distinct
// by default, and still targets CKKS when Sw == Sf.

// CHECK-CKKS: func.func @orbit_edge_rescale_placement
// CHECK-CKKS: secret.generic
// CHECK-CKKS: arith.mulf %input0, %input0
// CHECK-CKKS-NEXT: mgmt.relinearize
// CHECK-CKKS: mgmt.adjust_scale
// CHECK-CKKS: arith.addf
// CHECK-CKKS: mgmt.adjust_scale
// CHECK-CKKS-NEXT: mgmt.modreduce
// CHECK-CKKS-NOT: mgmt.bootstrap

// CHECK-CKKS-ALIGNED: func.func @orbit_edge_rescale_placement
// CHECK-CKKS-ALIGNED: secret.generic
// CHECK-CKKS-ALIGNED: arith.mulf %input0, %input0
// CHECK-CKKS-ALIGNED-NEXT: mgmt.relinearize
// CHECK-CKKS-ALIGNED: mgmt.modreduce
// CHECK-CKKS-ALIGNED-NOT: mgmt.bootstrap

!pt_ty = tensor<8xf32>
!ct_ty = !secret.secret<!pt_ty>

module attributes {scheme.ckks} {
  func.func @orbit_edge_rescale_placement(
      %arg0: !ct_ty)
      -> (!ct_ty {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 51>}) {
    %0 = secret.generic(%arg0: !ct_ty) {
    ^body(%input0: !pt_ty):
      %l1 = arith.mulf %input0, %input0 : !pt_ty
      %out = arith.addf %input0, %l1 : !pt_ty
      secret.yield %out : !pt_ty
    } -> (!ct_ty {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 51>})
    return %0 : !ct_ty
  }
}
