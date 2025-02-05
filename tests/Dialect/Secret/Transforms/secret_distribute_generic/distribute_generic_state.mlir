// RUN: heir-opt --secret-distribute-generic %s --split-input-file | FileCheck %s

// This tests a generic with multiple operations in its body, including a loop
// with secret loop-carried variables. For this test, the secretness analysis
// must have been re-run after each operation is split to ensure that the
// secretness state for a loop-carried variable is correct.

// CHECK-LABEL: test
// CHECK-SAME: %[[arg0:.*]]: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> {
module {
  func.func @test(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> {
    // CHECK-DAG:  %[[cst:.*]] = arith.constant
    // CHECK-DAG:  %[[v0:.*]] = secret.conceal %[[cst]] : tensor<1x1024xf32>
    // CHECK-NEXT: %[[v1:.*]] = affine.for %[[i:.*]] = 0 to 1023 iter_args(%[[arg2:.*]] = %[[v0]]) -> (!secret.secret<tensor<1x1024xf32>>) {
    // CHECK-NEXT:  %[[v3:.*]] = secret.generic ins(%[[arg0]], %[[arg2]] : !secret.secret<tensor<1x1024xf32>>, !secret.secret<tensor<1x1024xf32>>) {
    // CHECK-NEXT:  ^body(%[[input0:.*]]: tensor<1x1024xf32>, %[[input1:.*]]: tensor<1x1024xf32>):
    // CHECK-NEXT:    %[[v4:.*]] = arith.addf %[[input1]], %[[input0]] : tensor<1x1024xf32>
    // CHECK-NEXT:    secret.yield %[[v4]] : tensor<1x1024xf32>
    // CHECK-NEXT:  } -> !secret.secret<tensor<1x1024xf32>>
    // CHECK-NEXT:  affine.yield %[[v3]] : !secret.secret<tensor<1x1024xf32>>
    // CHECK-NEXT: }
    // CHECK-NEXT: %[[v2:.*]] = secret.generic ins(%[[v1]] : !secret.secret<tensor<1x1024xf32>>) {
    // CHECK-NEXT:  ^body(%[[input0:.*]]: tensor<1x1024xf32>):
    // CHECK-NEXT:    %[[v3:.*]] = arith.addf %[[input0]], %[[input0]] : tensor<1x1024xf32>
    // CHECK-NEXT:    secret.yield %[[v3]] : tensor<1x1024xf32>
    // CHECK-NEXT: } -> !secret.secret<tensor<1x1024xf32>>
    // CHECK-NEXT: return %[[v2]] : !secret.secret<tensor<1x1024xf32>>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3) -> (tensor<1x1024xf32>) {
        %9 = arith.addf %arg2, %input0 : tensor<1x1024xf32>
        affine.yield %9 : tensor<1x1024xf32>
      }
      %3 = arith.addf %1, %1 : tensor<1x1024xf32>
      secret.yield %3 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}

// -----

// CHECK-LABEL: test
// CHECK-SAME: %[[arg0:.*]]: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> {
module {
  func.func @test(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> {
    // CHECK-DAG:  %[[cst:.*]] = arith.constant
    // CHECK-NEXT: %[[v0:.*]] = secret.generic ins(%[[arg0]] : !secret.secret<tensor<1x1024xf32>>) {
    // CHECK-NEXT:  ^body(%[[input0:.*]]: tensor<1x1024xf32>):
    // CHECK-NEXT:    %[[v3:.*]] = arith.addf %[[input0]], %[[input0]] : tensor<1x1024xf32>
    // CHECK-NEXT:    secret.yield %[[v3]] : tensor<1x1024xf32>
    // CHECK-NEXT: } -> !secret.secret<tensor<1x1024xf32>>
    // CHECK-NEXT: %[[v1:.*]] = secret.conceal %[[cst]] : tensor<1x1024xf32>
    // CHECK-NEXT: %[[v2:.*]] = affine.for %[[i:.*]] = 0 to 1023 iter_args(%[[arg2:.*]] = %[[v1]]) -> (!secret.secret<tensor<1x1024xf32>>) {
    // CHECK-NEXT:  %[[v3:.*]] = secret.generic ins(%[[v0]], %[[arg2]] : !secret.secret<tensor<1x1024xf32>>, !secret.secret<tensor<1x1024xf32>>) {
    // CHECK-NEXT:  ^body(%[[input0:.*]]: tensor<1x1024xf32>, %[[input1:.*]]: tensor<1x1024xf32>):
    // CHECK-NEXT:    %[[v4:.*]] = arith.addf %[[input1]], %[[input0]] : tensor<1x1024xf32>
    // CHECK-NEXT:    secret.yield %[[v4]] : tensor<1x1024xf32>
    // CHECK-NEXT:  } -> !secret.secret<tensor<1x1024xf32>>
    // CHECK-NEXT:  affine.yield %[[v3]] : !secret.secret<tensor<1x1024xf32>>
    // CHECK-NEXT: }
    // CHECK-NEXT: return %[[v2]] : !secret.secret<tensor<1x1024xf32>>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1 = arith.addf %input0, %input0 : tensor<1x1024xf32>
      %2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3) -> (tensor<1x1024xf32>) {
        %9 = arith.addf %arg2, %1 : tensor<1x1024xf32>
        affine.yield %9 : tensor<1x1024xf32>
      }
      secret.yield %2 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}

// -----

// CHECK-LABEL: test
// CHECK-SAME: %[[arg0:.*]]: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> {
module {
  func.func @test(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> {
    // CHECK-DAG:  %[[cst:.*]] = arith.constant
    // CHECK-NEXT: %[[v0:.*]] = secret.generic ins(%[[arg0]] : !secret.secret<tensor<1x1024xf32>>) {
    // CHECK-NEXT:  ^body(%[[input0:.*]]: tensor<1x1024xf32>):
    // CHECK-NEXT:    %[[v3:.*]] = arith.addf %[[input0]], %[[input0]] : tensor<1x1024xf32>
    // CHECK-NEXT:    secret.yield %[[v3]] : tensor<1x1024xf32>
    // CHECK-NEXT: } -> !secret.secret<tensor<1x1024xf32>>
    // CHECK-NEXT: %[[v1:.*]] = secret.conceal %[[cst]] : tensor<1x1024xf32>
    // CHECK-NEXT: %[[v2:.*]] = affine.for %[[i:.*]] = 0 to 1023 iter_args(%[[arg2:.*]] = %[[v1]]) -> (!secret.secret<tensor<1x1024xf32>>) {
    // CHECK-NEXT:  %[[v3:.*]] = secret.generic ins(%[[v0]], %[[arg2]] : !secret.secret<tensor<1x1024xf32>>, !secret.secret<tensor<1x1024xf32>>) {
    // CHECK-NEXT:  ^body(%[[input0:.*]]: tensor<1x1024xf32>, %[[input1:.*]]: tensor<1x1024xf32>):
    // CHECK-NEXT:    %[[v4:.*]] = arith.addf %[[input1]], %[[input0]] : tensor<1x1024xf32>
    // CHECK-NEXT:    secret.yield %[[v4]] : tensor<1x1024xf32>
    // CHECK-NEXT:  } -> !secret.secret<tensor<1x1024xf32>>
    // CHECK-NEXT:  affine.yield %[[v3]] : !secret.secret<tensor<1x1024xf32>>
    // CHECK-NEXT: }
    // CHECK-NEXT: %[[v4:.*]] = secret.generic ins(%[[v2]] : !secret.secret<tensor<1x1024xf32>>) {
    // CHECK-NEXT:  ^body(%[[input0:.*]]: tensor<1x1024xf32>):
    // CHECK-NEXT:    %[[v3:.*]] = arith.addf %[[input0]], %[[input0]] : tensor<1x1024xf32>
    // CHECK-NEXT:    secret.yield %[[v3]] : tensor<1x1024xf32>
    // CHECK-NEXT: } -> !secret.secret<tensor<1x1024xf32>>
    // CHECK-NEXT: return %[[v4]] : !secret.secret<tensor<1x1024xf32>>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1 = arith.addf %input0, %input0 : tensor<1x1024xf32>
      %2 = affine.for %arg1 = 0 to 1023 iter_args(%arg2 = %cst_3) -> (tensor<1x1024xf32>) {
        %9 = arith.addf %arg2, %1 : tensor<1x1024xf32>
        affine.yield %9 : tensor<1x1024xf32>
      }
      %3 = arith.addf %2, %2 : tensor<1x1024xf32>
      secret.yield %3 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}
