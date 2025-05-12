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
}
