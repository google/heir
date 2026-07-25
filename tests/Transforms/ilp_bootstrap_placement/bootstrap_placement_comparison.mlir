// RUN: heir-opt --ilp-bootstrap-placement=bootstrap-waterline=3 %s | FileCheck %s --check-prefix=CHECK-ILP
// RUN: heir-opt --ilp-bootstrap-placement="bootstrap-waterline=3 orbit-cost-model=%S/orbit_cost_model.json" %s | FileCheck %s --check-prefix=CHECK-COST-MODEL
// RUN: heir-opt --secret-insert-mgmt-ckks=bootstrap-waterline=3 %s | FileCheck %s --check-prefix=CHECK-GREEDY

// Compare the greedy bootstrap placement against the ILP bootstrap placement
// with bootstrap-waterline=3. Greedy gives 3 bootstraps after L5, L6, L7. ILP
// places 2 bootstraps (optimal for this DAG with levels 0..3):
//
// Computation:
// L1 = I1 * I2   (level 2)
// L2 = I2 * I3   (level 2)
// L3 = I4 * I5   (level 2)
// L4 = L1 * L2   (level 1)
// L5 = I1 * L4   (would be level 0, needs bootstrap)
// L6 = L4 * I4   (would be level 0, needs bootstrap)
// L7 = L3 * L4   (would be level 0, needs bootstrap)
// L8 = L5 * L6   (would be level -1, needs bootstrap)
// L9 = L6 * L7   (would be level -1, needs bootstrap)
// L10 = L8 * L9  (would be level -2, needs bootstrap)

// CHECK-ILP-COUNT-2: mgmt.bootstrap
// CHECK-ILP-NOT: mgmt.bootstrap
// CHECK-GREEDY-COUNT-3: mgmt.bootstrap
// CHECK-COST-MODEL-COUNT-2: mgmt.bootstrap
// CHECK-COST-MODEL-NOT: mgmt.bootstrap

!pt_ty = tensor<8xf32>
!ct_ty = !secret.secret<!pt_ty>

func.func @bootstrap_placement_test(
    %arg0: !ct_ty, %arg1: !ct_ty, %arg2: !ct_ty, %arg3: !ct_ty, %arg4: !ct_ty) -> !ct_ty {
  %0 = secret.generic(
      %arg0: !ct_ty, %arg1: !ct_ty, %arg2: !ct_ty, %arg3: !ct_ty, %arg4: !ct_ty) {
  ^body(%input0: !pt_ty, %input1: !pt_ty, %input2: !pt_ty, %input3: !pt_ty, %input4: !pt_ty):
    // L1 = I1 * I2
    %l1 = arith.mulf %input0, %input1 : !pt_ty

    // L2 = I2 * I3
    %l2 = arith.mulf %input1, %input2 : !pt_ty

    // L3 = I4 * I5
    %l3 = arith.mulf %input3, %input4 : !pt_ty

    // L4 = L1 * L2
    %l4 = arith.mulf %l1, %l2 : !pt_ty

    // L5 = I1 * L4
    %l5 = arith.mulf %input0, %l4 : !pt_ty

    // L6 = L4 * I4
    %l6 = arith.mulf %l4, %input3 : !pt_ty

    // L7 = L3 * L4
    %l7 = arith.mulf %l3, %l4 : !pt_ty

    // L8 = L5 * L6
    %l8 = arith.mulf %l5, %l6 : !pt_ty

    // L9 = L6 * L7
    %l9 = arith.mulf %l6, %l7 : !pt_ty

    // L10 = L8 * L9
    %l10 = arith.mulf %l8, %l9 : !pt_ty

    secret.yield %l10 : !pt_ty
  } -> !ct_ty
  return %0 : !ct_ty
}
