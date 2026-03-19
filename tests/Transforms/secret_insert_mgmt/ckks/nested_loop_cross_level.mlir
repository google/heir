// RUN: heir-opt --secret-insert-mgmt-ckks="after-mul=true before-mul-include-first-mul=false bootstrap-waterline=10 level-budget=2 slot-number=1024" %s | FileCheck %s

// This test verifies that the secret-insert-mgmt-ckks pass correctly reconciles
// multiplicative level mismatches in nested loop structures. Specifically,
// when a bootstrapped outer-loop iteration argument is added to a reduced
// inner-loop result, the pass must insert a level_reduce on the bootstrapped
// operand to match the reduced level of the inner loop result.
//
// This was previously failing because MatchCrossLevel did not correctly handle
// the MaxLevel sentinel in the dataflow analysis for nested control flow.

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 10, Q = [36028797018652673, 35184372121601], P = [1152921504606994433], logDefaultScale = 45>, scheme.ckks} {
  // CHECK: func.func @nested_loop_cross_level
  func.func @nested_loop_cross_level(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> {
    %cst = arith.constant dense<1.000000e+00> : tensor<1x1024xf32>
    %c1 = arith.constant 1 : index
    %c23 = arith.constant 23 : index
    %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      // CHECK: scf.for
      %1 = scf.for %i = %c1 to %c23 step %c1 iter_args(%iter_outer = %cst) -> (tensor<1x1024xf32>) {
        // CHECK: scf.for
        %2 = scf.for %j = %c1 to %c23 step %c1 iter_args(%iter_inner = %cst) -> (tensor<1x1024xf32>) {
          %m = arith.mulf %input0, %iter_inner : tensor<1x1024xf32>
          scf.yield %m : tensor<1x1024xf32>
        }
        // CHECK: mgmt.level_reduce
        // CHECK: arith.addf
        %a = arith.addf %iter_outer, %2 : tensor<1x1024xf32>
        scf.yield %a : tensor<1x1024xf32>
      }
      secret.yield %1 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}
