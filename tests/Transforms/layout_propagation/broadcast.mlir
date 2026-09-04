// RUN: heir-opt --layout-propagation=min-slot-count=8192 %s | FileCheck %s

#bicyclic = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-2080i0 - 66i1 + slot) mod 2145 = 0 and 0 <= i0 <= 32 and 0 <= i1 <= 64 and 0 <= slot <= 8191 }">
#row_major = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-4i0 - i1 + slot) mod 1024 = 0 and 0 <= i0 <= 2 and 0 <= i1 <= 3 and 0 <= slot <= 8191 }">

// CHECK-DAG: #[[RES_LAYOUT:.*]] = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : ct = 0 and (2145i0 - 2080i1 - 66i2 + slot) mod 4290 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 32 and 0 <= i2 <= 64 and 0 <= slot <= 8191 }">
// CHECK-DAG: #[[RES_RM_LAYOUT:.*]] = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : ct = 0 and (-12i0 - 4i1 - i2 + slot) mod {{.*}} }">

module {
  // CHECK: @broadcast_2d_to_3d
  // CHECK: linalg.broadcast
  // CHECK-SAME: dimensions = [0]
  // CHECK-SAME: tensor_ext.layout = #[[RES_LAYOUT]]
  // CHECK-NOT: tensor_ext.convert_layout
  // CHECK: return
  func.func @broadcast_2d_to_3d(%arg0: !secret.secret<tensor<33x65xf32>> {tensor_ext.layout = #bicyclic}) -> !secret.secret<tensor<2x33x65xf32>> {
    %init = arith.constant dense<0.000000e+00> : tensor<2x33x65xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<33x65xf32>>) {
    ^body(%input0: tensor<33x65xf32>):
      %broadcasted = linalg.broadcast ins(%input0 : tensor<33x65xf32>) outs(%init : tensor<2x33x65xf32>) dimensions = [0]
      secret.yield %broadcasted : tensor<2x33x65xf32>
    } -> !secret.secret<tensor<2x33x65xf32>>
    return %0 : !secret.secret<tensor<2x33x65xf32>>
  }

  // CHECK: @broadcast_not_coprime
  // CHECK: linalg.broadcast
  // CHECK-SAME: dimensions = [0]
  // CHECK-SAME: tensor_ext.layout = #[[RES_RM_LAYOUT]]
  // CHECK-NOT: tensor_ext.convert_layout
  // CHECK: return
  func.func @broadcast_not_coprime(%arg0: !secret.secret<tensor<3x4xf32>> {tensor_ext.layout = #row_major}) -> !secret.secret<tensor<2x3x4xf32>> {
    %init = arith.constant dense<0.000000e+00> : tensor<2x3x4xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<3x4xf32>>) {
    ^body(%input0: tensor<3x4xf32>):
      %broadcasted = linalg.broadcast ins(%input0 : tensor<3x4xf32>) outs(%init : tensor<2x3x4xf32>) dimensions = [0]
      secret.yield %broadcasted : tensor<2x3x4xf32>
    } -> !secret.secret<tensor<2x3x4xf32>>
    return %0 : !secret.secret<tensor<2x3x4xf32>>
  }
}
