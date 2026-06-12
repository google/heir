// RUN: heir-opt --ilp-bootstrap-placement=bootstrap-waterline=3 %s | FileCheck %s

// Without CKKS module attributes, the ILP uses generic level-only management
// with equal effective Sw/Sf. This should not require callers to manually set
// the scale options equal.

// CHECK: func.func @orbit_level_only_placement
// CHECK: secret.generic
// CHECK: arith.mulf %input0, %input0
// CHECK-NEXT: mgmt.relinearize
// CHECK-NEXT: mgmt.level_reduce
// CHECK-NOT: mgmt.adjust_scale
// CHECK-NOT: mgmt.modreduce

!pt_ty = tensor<8xf32>
!ct_ty = !secret.secret<!pt_ty>

func.func @orbit_level_only_placement(%arg0: !ct_ty) -> !ct_ty {
  %0 = secret.generic(%arg0: !ct_ty) {
  ^body(%input0: !pt_ty):
    %l1 = arith.mulf %input0, %input0 : !pt_ty
    %out = arith.addf %input0, %l1 : !pt_ty
    secret.yield %out : !pt_ty
  } -> !ct_ty
  return %0 : !ct_ty
}
