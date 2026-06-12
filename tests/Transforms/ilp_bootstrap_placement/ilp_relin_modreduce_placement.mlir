// RUN: heir-opt --ilp-bootstrap-placement=bootstrap-waterline=3 %s | FileCheck %s

// Test that relinearize is inserted after multiplications and that modreduce is
// only inserted at rescale positions chosen by the ILP, not unconditionally
// after every multiplication.

// CHECK: func.func @bootstrap_placement_test
// CHECK: secret.generic
// CHECK: arith.mulf %input0, %input1
// CHECK-NEXT: mgmt.relinearize
// CHECK-NEXT: arith.mulf %input1, %input2
// CHECK: mgmt.modreduce
// CHECK: mgmt.bootstrap
// CHECK: secret.yield

!pt_ty = tensor<8xf32>
!ct_ty = !secret.secret<!pt_ty>

module attributes {scheme.ckks} {
  func.func @bootstrap_placement_test(
      %arg0: !ct_ty, %arg1: !ct_ty, %arg2: !ct_ty, %arg3: !ct_ty, %arg4: !ct_ty) -> !ct_ty {
    %0 = secret.generic(
        %arg0: !ct_ty, %arg1: !ct_ty, %arg2: !ct_ty, %arg3: !ct_ty, %arg4: !ct_ty) {
    ^body(%input0: !pt_ty, %input1: !pt_ty, %input2: !pt_ty, %input3: !pt_ty, %input4: !pt_ty):
      %l1 = arith.mulf %input0, %input1 : !pt_ty
      %l2 = arith.mulf %input1, %input2 : !pt_ty
      %l3 = arith.mulf %input3, %input4 : !pt_ty
      %l4 = arith.mulf %l1, %l2 : !pt_ty
      %l5 = arith.mulf %input0, %l4 : !pt_ty
      %l6 = arith.mulf %l4, %input3 : !pt_ty
      %l7 = arith.mulf %l3, %l4 : !pt_ty
      %l8 = arith.mulf %l5, %l6 : !pt_ty
      %l9 = arith.mulf %l6, %l7 : !pt_ty
      %l10 = arith.mulf %l8, %l9 : !pt_ty
      secret.yield %l10 : !pt_ty
    } -> !ct_ty
    return %0 : !ct_ty
  }
}
