// RUN: heir-opt --ilp-bootstrap-placement=bootstrap-waterline=3 %s | FileCheck %s
// RUN: heir-opt --ilp-bootstrap-placement="bootstrap-waterline=3 partition-min-size=1" %s | FileCheck %s
// RUN: heir-opt --ilp-bootstrap-placement="bootstrap-waterline=3 partition-min-size=1 compress=false" %s | FileCheck %s

// A pure squaring chain has a single-input single-output cut after every
// mul, so partition-min-size=1 solves each op as its own partition and
// stitches the boundary levels by dynamic programming. The stitched solution
// must match the single-partition optimum: five squarings from level 3 need
// exactly one bootstrap.

// CHECK: func.func @siso_partition_chain
// CHECK-COUNT-1: mgmt.bootstrap
// CHECK-NOT: mgmt.bootstrap

!pt_ty = tensor<8xf32>
!ct_ty = !secret.secret<!pt_ty>

func.func @siso_partition_chain(%arg0: !ct_ty) -> !ct_ty {
  %0 = secret.generic(%arg0: !ct_ty) {
  ^body(%input0: !pt_ty):
    %m1 = arith.mulf %input0, %input0 : !pt_ty
    %m2 = arith.mulf %m1, %m1 : !pt_ty
    %m3 = arith.mulf %m2, %m2 : !pt_ty
    %m4 = arith.mulf %m3, %m3 : !pt_ty
    %m5 = arith.mulf %m4, %m4 : !pt_ty
    secret.yield %m5 : !pt_ty
  } -> !ct_ty
  return %0 : !ct_ty
}
