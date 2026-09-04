// RUN: heir-opt --layout-propagation=min-slot-count=8192 --convert-to-ciphertext-semantics=min-slot-count=8192 %s | FileCheck %s

#bicyclic = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (65i0 - 66i1 + slot) mod 2145 = 0 and 0 <= i0 <= 32 and 0 <= i1 <= 64 and 0 <= slot <= 8191 }">

// CHECK: func.func @broadcast_2d_to_3d
func.func @broadcast_2d_to_3d(%arg0: !secret.secret<tensor<33x65xf32>> {tensor_ext.layout = #bicyclic}) -> !secret.secret<tensor<2x33x65xf32>> {
  %init = arith.constant dense<0.000000e+00> : tensor<2x33x65xf32>
  // CHECK: secret.generic
  // CHECK-NEXT: ^body(%[[IN:.*]]: tensor<1x8192xf32>):
  // CHECK: secret.yield %[[IN]] : tensor<1x8192xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<33x65xf32>>) {
  ^body(%input0: tensor<33x65xf32>):
    %broadcasted = linalg.broadcast ins(%input0 : tensor<33x65xf32>) outs(%init : tensor<2x33x65xf32>) dimensions = [0]
    secret.yield %broadcasted : tensor<2x33x65xf32>
  } -> !secret.secret<tensor<2x33x65xf32>>
  return %0 : !secret.secret<tensor<2x33x65xf32>>
}
