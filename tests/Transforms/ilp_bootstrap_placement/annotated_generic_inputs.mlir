// RUN: heir-opt --ilp-bootstrap-placement=bootstrap-waterline=3 %s | FileCheck %s --check-prefix=CHECK-LEVEL

!pt_ty = tensor<8xf32>
!ct_ty = !secret.secret<!pt_ty>

// CHECK-LEVEL: func.func @uses_annotated_input_level
// CHECK-LEVEL: secret.generic
// CHECK-LEVEL: arith.mulf
// CHECK-LEVEL: mgmt.bootstrap
// CHECK-LEVEL: arith.mulf
func.func @uses_annotated_input_level(
    %arg0: !ct_ty {mgmt.mgmt = #mgmt.mgmt<level = 1>}) -> !ct_ty {
  %0 = secret.generic(
      %arg0: !ct_ty {mgmt.mgmt = #mgmt.mgmt<level = 1>}) {
  ^body(%input0: !pt_ty):
    %l1 = arith.mulf %input0, %input0 : !pt_ty
    %l2 = arith.mulf %l1, %input0 : !pt_ty
    secret.yield %l2 : !pt_ty
  } -> !ct_ty
  return %0 : !ct_ty
}
