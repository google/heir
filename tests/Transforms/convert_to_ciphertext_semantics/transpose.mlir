// RUN: heir-opt --layout-propagation=min-slot-count=8192 --convert-to-ciphertext-semantics=min-slot-count=8192 %s | FileCheck %s

#bicyclic = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-66i0 + 65i1 + slot) mod 2145 = 0 and 0 <= i0 <= 32 and 0 <= i1 <= 64 and 0 <= slot <= 8191 }">

// CHECK: func.func @transpose_2d
func.func @transpose_2d(%arg0: !secret.secret<tensor<33x65xf32>> {tensor_ext.layout = #bicyclic}) -> !secret.secret<tensor<65x33xf32>> {
  %init = arith.constant dense<0.000000e+00> : tensor<65x33xf32>
  // CHECK: secret.generic
  // CHECK-NEXT: ^body(%[[IN:.*]]: tensor<1x8192xf32>):
  // CHECK: secret.yield %[[IN]] : tensor<1x8192xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<33x65xf32>>) {
  ^body(%input0: tensor<33x65xf32>):
    %transposed = linalg.transpose ins(%input0 : tensor<33x65xf32>) outs(%init : tensor<65x33xf32>) permutation = [1, 0]
    secret.yield %transposed : tensor<65x33xf32>
  } -> !secret.secret<tensor<65x33xf32>>
  return %0 : !secret.secret<tensor<65x33xf32>>
}
