// RUN: heir-opt --canonicalize %s | FileCheck %s

// Ensures that a plaintext tensor is concealed if it has secret values inserted.

module {
  // CHECK: func.func @main
  // CHECK-SAME: (%[[arg0:.*]]: !secret.secret<i32>)
  // CHECK: %[[cst:.*]] = arith.constant dense<1> : tensor<2xi32>
  // CHECK: %[[v0:.*]] = secret.conceal %[[cst]]
  // CHECK: secret.generic
  // CHECK-SAME: (%[[arg0]]: !secret.secret<i32>, %[[v0]]: !secret.secret<tensor<2xi32>>)
  // CHECK-NEXT: ^body(%[[input0:.*]]: i32, %[[input1:.*]]: tensor<2xi32>)
  // CHECK: tensor.insert %[[input0]] into %[[input1]]
  func.func @main(%arg0: !secret.secret<i32>) -> !secret.secret<tensor<2xi32>> {
    %0 = secret.generic(%arg0: !secret.secret<i32>) {
    ^body(%input0: i32):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant dense<1> : tensor<2xi32>
      %inserted = tensor.insert %input0 into %cst[%c0] : tensor<2xi32>
      %inserted_1 = tensor.insert %input0 into %inserted[%c1] : tensor<2xi32>
      secret.yield %inserted_1 : tensor<2xi32>
    } -> !secret.secret<tensor<2xi32>>
    return %0 : !secret.secret<tensor<2xi32>>
  }

  // When a plaintext tensor is concealed for an insertion inside the generic,
  // an insertion into it that was already hoisted out of the generic keeps
  // using the plaintext tensor.
  // CHECK: func.func @hoisted_insert
  // CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : tensor<4xf64>
  // CHECK: tensor.insert %{{.*}} into %[[cst]]
  // CHECK: secret.generic
  func.func @hoisted_insert(%arg0: tensor<16xf64>, %arg1: !secret.secret<tensor<4xf64>>) -> !secret.secret<tensor<4xf64>> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant dense<0.0> : tensor<4xf64>
    %0 = secret.generic(%arg1: !secret.secret<tensor<4xf64>>) {
    ^body(%x: tensor<4xf64>):
      %e0 = tensor.extract %arg0[%c0] : tensor<16xf64>
      %i0 = tensor.insert %e0 into %cst[%c0] : tensor<4xf64>
      %e1 = tensor.extract %arg0[%c1] : tensor<16xf64>
      %i1 = tensor.insert %e1 into %cst[%c0] : tensor<4xf64>
      %s = arith.addf %i0, %i1 : tensor<4xf64>
      %r = arith.mulf %s, %x : tensor<4xf64>
      secret.yield %r : tensor<4xf64>
    } -> !secret.secret<tensor<4xf64>>
    return %0 : !secret.secret<tensor<4xf64>>
  }
}
