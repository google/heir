// RUN: heir-opt --annotate-level %s | FileCheck %s

module {
  func.func @loop_matvec(%arg0: !secret.secret<tensor<16xf32>>) -> !secret.secret<tensor<16xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant dense<1.100000e+00> : tensor<16xf32>

    %0 = secret.generic(%arg0 : !secret.secret<tensor<16xf32>>) {
    ^body(%val: tensor<16xf32>):
      // CHECK: scf.for
      %1 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %cst) -> (tensor<16xf32>) {
        // CHECK: arith.mulf
        // CHECK-SAME: {mgmt.level = 0 : index}
        %2 = arith.mulf %val, %cst_0 : tensor<16xf32>
        // CHECK: mgmt.modreduce
        // CHECK-SAME: {mgmt.level = 1 : index}
        %3 = mgmt.modreduce %2 : tensor<16xf32>
        // CHECK: arith.addf
        // CHECK-SAME: {mgmt.level = 1 : index}
        %4 = arith.addf %arg2, %3 : tensor<16xf32>
        scf.yield %4 : tensor<16xf32>
      }
      secret.yield %1 : tensor<16xf32>
    } -> !secret.secret<tensor<16xf32>>
    return %0 : !secret.secret<tensor<16xf32>>
  }
}
