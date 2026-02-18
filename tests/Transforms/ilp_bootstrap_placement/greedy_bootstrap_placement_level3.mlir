// RUN: heir-opt --secret-insert-mgmt-ckks=bootstrap-waterline=3 %s | FileCheck %s

// Test greedy bootstrap placement with bootstrap-waterline=3
// (bootstrap-waterline=3 semantics: bootstrap when level hits 0). Greedy
// gives 3 bootstraps after L5, L6, L7 (each bootstrap immediately after the
// modreduce that brings that branch to level 0).
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


// CHECK: func.func @bootstrap_placement_test
// CHECK: secret.generic
// CHECK-DAG: arith.mulf
// Bootstrap after L5 (modreduce %16 is L5 chain)
// CHECK: mgmt.modreduce %16
// CHECK-NEXT: mgmt.bootstrap %27
// Bootstrap after L6 (modreduce %21 is L6 chain)
// CHECK: mgmt.modreduce %21
// CHECK-NEXT: mgmt.bootstrap %29
// Bootstrap after L7 (modreduce %26 is L7 chain)
// CHECK: mgmt.modreduce %26
// CHECK-NEXT: mgmt.bootstrap %33
// CHECK: secret.yield

func.func @bootstrap_placement_test(
    %arg0: !secret.secret<tensor<8xf32>>,
    %arg1: !secret.secret<tensor<8xf32>>,
    %arg2: !secret.secret<tensor<8xf32>>,
    %arg3: !secret.secret<tensor<8xf32>>,
    %arg4: !secret.secret<tensor<8xf32>>) -> !secret.secret<tensor<8xf32>> {
  %0 = secret.generic(
      %arg0: !secret.secret<tensor<8xf32>>,
      %arg1: !secret.secret<tensor<8xf32>>,
      %arg2: !secret.secret<tensor<8xf32>>,
      %arg3: !secret.secret<tensor<8xf32>>,
      %arg4: !secret.secret<tensor<8xf32>>) {
  ^body(%input0: tensor<8xf32>,
        %input1: tensor<8xf32>,
        %input2: tensor<8xf32>,
        %input3: tensor<8xf32>,
        %input4: tensor<8xf32>):
    // L1 = I1 * I2
    %l1 = arith.mulf %input0, %input1 : tensor<8xf32>

    // L2 = I2 * I3
    %l2 = arith.mulf %input1, %input2 : tensor<8xf32>

    // L3 = I4 * I5
    %l3 = arith.mulf %input3, %input4 : tensor<8xf32>

    // L4 = L1 * L2
    %l4 = arith.mulf %l1, %l2 : tensor<8xf32>

    // L5 = I1 * L4
    %l5 = arith.mulf %input0, %l4 : tensor<8xf32>

    // L6 = L4 * I4
    %l6 = arith.mulf %l4, %input3 : tensor<8xf32>

    // L7 = L3 * L4
    %l7 = arith.mulf %l3, %l4 : tensor<8xf32>

    // L8 = L5 * L6
    %l8 = arith.mulf %l5, %l6 : tensor<8xf32>

    // L9 = L6 * L7
    %l9 = arith.mulf %l6, %l7 : tensor<8xf32>

    // L10 = L8 * L9
    %l10 = arith.mulf %l8, %l9 : tensor<8xf32>

    secret.yield %l10 : tensor<8xf32>
  } -> !secret.secret<tensor<8xf32>>
  return %0 : !secret.secret<tensor<8xf32>>
}
