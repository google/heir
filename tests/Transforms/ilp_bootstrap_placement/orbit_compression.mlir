// RUN: heir-opt --ilp-bootstrap-placement=bootstrap-waterline=2 %s | FileCheck %s
// RUN: heir-opt --ilp-bootstrap-placement="bootstrap-waterline=2 compress=false" %s | FileCheck %s

// Compression must not change the chosen placement on this circuit: the
// three parallel muls of (%s, %input1) are structurally identical and merge
// into one compression class, and the addition chain squashes into one group. With
// bootstrap-waterline=2 the chain runs out of levels and exactly one
// bootstrap is optimal; a bootstrap on the merged three-mul class would be
// charged (and decoded) three times, so the solver avoids it. Both the
// compressed and uncompressed models produce identical IR.

// CHECK: func.func @compression_invariance
// CHECK-COUNT-1: mgmt.bootstrap
// CHECK-NOT: mgmt.bootstrap

!pt_ty = tensor<8xf32>
!ct_ty = !secret.secret<!pt_ty>

func.func @compression_invariance(
    %arg0: !ct_ty, %arg1: !ct_ty) -> !ct_ty {
  %0 = secret.generic(%arg0: !ct_ty, %arg1: !ct_ty) {
  ^body(%input0: !pt_ty, %input1: !pt_ty):
    %s = arith.mulf %input0, %input0 : !pt_ty
    // Three structurally identical ops: one compression class of weight 3.
    %p1 = arith.mulf %s, %input1 : !pt_ty
    %p2 = arith.mulf %s, %input1 : !pt_ty
    %p3 = arith.mulf %s, %input1 : !pt_ty
    // One addition tree: one squashed group.
    %a1 = arith.addf %p1, %p2 : !pt_ty
    %a2 = arith.addf %a1, %p3 : !pt_ty
    %out = arith.mulf %a2, %a2 : !pt_ty
    secret.yield %out : !pt_ty
  } -> !ct_ty
  return %0 : !ct_ty
}
