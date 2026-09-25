// RUN: heir-opt --ilp-bootstrap-placement="bootstrap-waterline=7 partition-min-size=1 bypass-depth-threshold=3" %s | FileCheck %s --implicit-check-not=mgmt.bootstrap

// A residual block sitting mid-circuit: a shared prefix %p forks into a
// multiplicative-depth-4 main chain and an identity skip, rejoined at an
// addition whose result feeds a suffix multiply. With partition-min-size=1 the
// body splits into [prefix] [residual] [suffix]; the residual partition has no
// single-input single-output cut, so it is bypass-solved inside the partition
// dynamic program (the fork %p is pinned to each boundary state the DP
// explores). Waterline 7 covers the depth-6 flow, so no bootstrap is needed.

// CHECK: func.func @nested_residual
// CHECK-COUNT-6: arith.mulf
// CHECK: secret.yield

!pt_ty = tensor<8xf32>
!ct_ty = !secret.secret<!pt_ty>

func.func @nested_residual(%arg0: !ct_ty) -> !ct_ty {
  %0 = secret.generic(%arg0: !ct_ty) {
  ^body(%in: !pt_ty):
    %p = arith.mulf %in, %in : !pt_ty
    %m1 = arith.mulf %p, %p : !pt_ty
    %m2 = arith.mulf %m1, %m1 : !pt_ty
    %m3 = arith.mulf %m2, %m2 : !pt_ty
    %m4 = arith.mulf %m3, %m3 : !pt_ty
    %out = arith.addf %m4, %p : !pt_ty
    %s = arith.mulf %out, %out : !pt_ty
    secret.yield %s : !pt_ty
  } -> !ct_ty
  return %0 : !ct_ty
}
