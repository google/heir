// RUN: heir-opt --annotate-level="level-budget=2" %s | FileCheck %s

module {
  func.func @level_growth(%arg0: !secret.secret<tensor<16xf32>>) -> !secret.secret<tensor<16xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index

    %0 = secret.generic(%arg0 : !secret.secret<tensor<16xf32>>) {
    ^body(%val: tensor<16xf32>):
      // CHECK: scf.for
      %1 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %val) -> (tensor<16xf32>) {
        // CHECK: mgmt.modreduce
        // CHECK-SAME: {mgmt.level = "invalid"}
        %2 = mgmt.modreduce %arg2 : tensor<16xf32>
        scf.yield %2 : tensor<16xf32>
      }
      secret.yield %1 : tensor<16xf32>
    } -> !secret.secret<tensor<16xf32>>
    return %0 : !secret.secret<tensor<16xf32>>
  }
}
