// RUN: heir-opt --layout-propagation=min-slot-count=8192 %s | FileCheck %s

#bicyclic = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-66i0 + 65i1 + slot) mod 2145 = 0 and 0 <= i0 <= 32 and 0 <= i1 <= 64 and 0 <= slot <= 8191 }">
#tricyclic = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : ct = 0 and (2145i0 - 2080i1 - 66i2 + slot) mod 4290 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 32 and 0 <= i2 <= 64 and 0 <= slot <= 8191 }">

// CHECK-DAG: #[[LAYOUT_2D:.*]] = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (65i0 - 66i1 + slot) mod 2145 = 0 and 0 <= i0 <= 64 and 0 <= i1 <= 32 and 0 <= slot <= 8191 }">
// CHECK-DAG: #[[LAYOUT_3D:.*]] = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : ct = 0 and (-2080i0 + 2145i1 - 66i2 + slot) mod 4290 = 0 and 0 <= i0 <= 32 and 0 <= i1 <= 1 and 0 <= i2 <= 64 and 0 <= slot <= 8191 }">

module {
  // CHECK: func.func @transpose_2d
  // CHECK: linalg.transpose
  // CHECK-SAME: permutation = [1, 0]
  // CHECK-SAME: tensor_ext.layout = #[[LAYOUT_2D]]
  // CHECK: return
  func.func @transpose_2d(%arg0: !secret.secret<tensor<33x65xf32>> {tensor_ext.layout = #bicyclic}) -> !secret.secret<tensor<65x33xf32>> {
    %init = arith.constant dense<0.000000e+00> : tensor<65x33xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<33x65xf32>>) {
    ^body(%input0: tensor<33x65xf32>):
      %transposed = linalg.transpose ins(%input0 : tensor<33x65xf32>) outs(%init : tensor<65x33xf32>) permutation = [1, 0]
      secret.yield %transposed : tensor<65x33xf32>
    } -> !secret.secret<tensor<65x33xf32>>
    return %0 : !secret.secret<tensor<65x33xf32>>
  }

  // CHECK: func.func @transpose_3d
  // CHECK: linalg.transpose
  // CHECK-SAME: permutation = [1, 0, 2]
  // CHECK-SAME: tensor_ext.layout = #[[LAYOUT_3D]]
  // CHECK: return
  func.func @transpose_3d(%arg0: !secret.secret<tensor<2x33x65xf32>> {tensor_ext.layout = #tricyclic}) -> !secret.secret<tensor<33x2x65xf32>> {
    %init = arith.constant dense<0.000000e+00> : tensor<33x2x65xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<2x33x65xf32>>) {
    ^body(%input0: tensor<2x33x65xf32>):
      %transposed = linalg.transpose ins(%input0 : tensor<2x33x65xf32>) outs(%init : tensor<33x2x65xf32>) permutation = [1, 0, 2]
      secret.yield %transposed : tensor<33x2x65xf32>
    } -> !secret.secret<tensor<33x2x65xf32>>
    return %0 : !secret.secret<tensor<33x2x65xf32>>
  }
}
