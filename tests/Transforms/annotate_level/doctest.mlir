// RUN: heir-opt --annotate-level="level-budget=5" %s | FileCheck %s --check-prefix=CHECK-B5

module {
  // Part 1: No loop, at least three levels
  func.func @no_loop(%arg0: !secret.secret<tensor<16xf32>>) -> !secret.secret<tensor<16xf32>> {
    %0 = secret.generic(%arg0 : !secret.secret<tensor<16xf32>>) {
    ^body(%val: tensor<16xf32>):
      // CHECK-B5: mgmt.modreduce
      // CHECK-B5-SAME: {mgmt.level = 1 : index}
      %1 = mgmt.modreduce %val : tensor<16xf32>

      // CHECK-B5: mgmt.modreduce
      // CHECK-B5-SAME: {mgmt.level = 2 : index}
      %2 = mgmt.modreduce %1 : tensor<16xf32>

      // CHECK-B5: mgmt.modreduce
      // CHECK-B5-SAME: {mgmt.level = 3 : index}
      %3 = mgmt.modreduce %2 : tensor<16xf32>

      secret.yield %3 : tensor<16xf32>
    } -> !secret.secret<tensor<16xf32>>
    return %0 : !secret.secret<tensor<16xf32>>
  }

  // Part 2: Loop that converges
  func.func @loop_converges(%arg0: !secret.secret<tensor<16xf32>>) -> !secret.secret<tensor<16xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant dense<1.100000e+00> : tensor<16xf32>

    %0 = secret.generic(%arg0 : !secret.secret<tensor<16xf32>>) {
    ^body(%val: tensor<16xf32>):
      %1 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %cst) -> (tensor<16xf32>) {
        %2 = arith.mulf %val, %cst_0 : tensor<16xf32>
        // CHECK-B5: mgmt.modreduce
        // CHECK-B5-SAME: {mgmt.level = 1 : index}
        %3 = mgmt.modreduce %2 : tensor<16xf32>

        // CHECK-B5: arith.addf
        // CHECK-B5-SAME: {mgmt.level = 1 : index}
        %4 = arith.addf %arg2, %3 : tensor<16xf32>
        scf.yield %4 : tensor<16xf32>
      }
      secret.yield %1 : tensor<16xf32>
    } -> !secret.secret<tensor<16xf32>>
    return %0 : !secret.secret<tensor<16xf32>>
  }

  // Part 3: Loop that does not converge (grows to 10)
  func.func @loop_diverges(%arg0: !secret.secret<tensor<16xf32>>) -> !secret.secret<tensor<16xf32>> {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index

    %0 = secret.generic(%arg0 : !secret.secret<tensor<16xf32>>) {
    ^body(%val: tensor<16xf32>):
      // CHECK-B5: scf.for
      %1 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %val) -> (tensor<16xf32>) {
        // CHECK-B5: mgmt.modreduce
        // CHECK-B5-SAME: {mgmt.level = "invalid"}
        %2 = mgmt.modreduce %arg2 : tensor<16xf32>
        scf.yield %2 : tensor<16xf32>
      }
      secret.yield %1 : tensor<16xf32>
    } -> !secret.secret<tensor<16xf32>>
    return %0 : !secret.secret<tensor<16xf32>>
  }
}
