// RUN: heir-opt --layout-propagation=min-slot-count=1024 --convert-to-ciphertext-semantics=min-slot-count=1024 %s | FileCheck %s

#bicyclic = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-10i0 - 6i1 + slot) mod 15 = 0 and 0 <= i0 <= 2 and 0 <= i1 <= 4 and 0 <= slot <= 1023 }">
#tricyclic_2_3_5 = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : ct = 0 and (-15i0 - 10i1 - 6i2 + slot) mod 30 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 2 and 0 <= i2 <= 4 and 0 <= slot <= 1023 }">
#tricyclic_2_5_7 = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : ct = 0 and (-35i0 - 56i1 - 50i2 + slot) mod 70 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 4 and 0 <= i2 <= 6 and 0 <= slot <= 1023 }">

module {
  // CHECK: @batch_matmul_broadcast
  // CHECK-NOT: linalg.batch_matmul
  // CHECK: tensor_ext.rotate
  // CHECK: arith.mulf
  // CHECK: tensor_ext.remap
  func.func @batch_matmul_broadcast(%arg0: !secret.secret<tensor<3x5xf32>> {tensor_ext.layout = #bicyclic}, %arg1: tensor<2x5x7xf32>) -> !secret.secret<tensor<2x3x7xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x3x7xf32>
    %empty = tensor.empty() : tensor<2x3x5xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<3x5xf32>> {tensor_ext.layout = #bicyclic}) {
    ^body(%input0: tensor<3x5xf32>):
      %bcast = linalg.broadcast ins(%input0 : tensor<3x5xf32>) outs(%empty : tensor<2x3x5xf32>) dimensions = [0]
      %1 = linalg.batch_matmul ins(%bcast, %arg1 : tensor<2x3x5xf32>, tensor<2x5x7xf32>) outs(%cst : tensor<2x3x7xf32>) -> tensor<2x3x7xf32>
      secret.yield %1 : tensor<2x3x7xf32>
    } -> !secret.secret<tensor<2x3x7xf32>>
    return %0 : !secret.secret<tensor<2x3x7xf32>>
  }

  // CHECK: @batch_matmul_ctpt
  // CHECK-NOT: linalg.batch_matmul
  // CHECK: tensor_ext.rotate
  // CHECK: arith.mulf
  // CHECK: tensor_ext.remap
  func.func @batch_matmul_ctpt(%arg0: !secret.secret<tensor<2x3x5xf32>> {tensor_ext.layout = #tricyclic_2_3_5}, %arg1: tensor<2x5x7xf32>) -> !secret.secret<tensor<2x3x7xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x3x7xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<2x3x5xf32>> {tensor_ext.layout = #tricyclic_2_3_5}) {
    ^body(%input0: tensor<2x3x5xf32>):
      %1 = linalg.batch_matmul ins(%input0, %arg1 : tensor<2x3x5xf32>, tensor<2x5x7xf32>) outs(%cst : tensor<2x3x7xf32>) -> tensor<2x3x7xf32>
      secret.yield %1 : tensor<2x3x7xf32>
    } -> !secret.secret<tensor<2x3x7xf32>>
    return %0 : !secret.secret<tensor<2x3x7xf32>>
  }

  // CHECK: @batch_matmul_ptct
  // CHECK-NOT: linalg.batch_matmul
  // CHECK: tensor_ext.rotate
  // CHECK: arith.mulf
  // CHECK: tensor_ext.remap
  func.func @batch_matmul_ptct(%arg0: tensor<2x3x5xf32>, %arg1: !secret.secret<tensor<2x5x7xf32>> {tensor_ext.layout = #tricyclic_2_5_7}) -> !secret.secret<tensor<2x3x7xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x3x7xf32>
    %0 = secret.generic(%arg1: !secret.secret<tensor<2x5x7xf32>> {tensor_ext.layout = #tricyclic_2_5_7}) {
    ^body(%input0: tensor<2x5x7xf32>):
      %1 = linalg.batch_matmul ins(%arg0, %input0 : tensor<2x3x5xf32>, tensor<2x5x7xf32>) outs(%cst : tensor<2x3x7xf32>) -> tensor<2x3x7xf32>
      secret.yield %1 : tensor<2x3x7xf32>
    } -> !secret.secret<tensor<2x3x7xf32>>
    return %0 : !secret.secret<tensor<2x3x7xf32>>
  }
}
