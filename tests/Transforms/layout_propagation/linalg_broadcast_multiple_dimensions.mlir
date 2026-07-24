// RUN: heir-opt --layout-propagation --split-input-file %s | FileCheck %s
// This tests the broadcast swap on a linalg broadcast

// CHECK-DAG: #layout = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : (-64i1 - i2 + slot + 1024*floor((i1)/16)) mod 262144 = 0 and 0 <= i0 <= 63 and 0 <= i1 <= 63 and 0 <= i2 <= 63 and 0 <= ct <= 255 and -15 + 64i0 + i1 <= 16ct <= 64i0 + i1 and 0 <= slot <= 1023 }">
// CHECK-DAG: #layout1 = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 4i0 and slot = 0 and 0 <= i0 <= 63 }">
module {
  // CHECK: func @main
  func.func @main(%arg0: !secret.secret<tensor<64x64x64xf32>>) -> (!secret.secret<tensor<64x64x64xf32>>) {
    %cst = arith.constant dense<0.0> : tensor<64xf32>
    %cst_0 = arith.constant dense<2.0> : tensor<64xf32>
    %0 = tensor.empty() {secret.public} : tensor<64x64x64xf32>
    %1 = secret.generic(%arg0: !secret.secret<tensor<64x64x64xf32>>) {
    ^body(%input0: tensor<64x64x64xf32>):
      // CHECK-DAG linalg.reduce ins(%input0 : tensor<64x64x64xf32>) outs(%cst : tensor<64xf32>) dimensions = [1,2] {tensor_ext.layout = [[#layout]]}
      %reduced = linalg.reduce ins(%input0 : tensor<64x64x64xf32>) outs(%cst : tensor<64xf32>) dimensions = [1,2]
        (%in: f32, %init: f32) {
          %31 = arith.addf %in, %init {secret.public} : f32
          linalg.yield %31 {secret.public} : f32
        }
      // CHECK-DAG arith.mulf %reduced, %cst_0 : tensor<64xf32> {tensor_ext.layout = [[#layout1]]}
      %2 = arith.mulf %reduced, %cst_0 : tensor<64xf32>
      // CHECK-DAG linalg.broadcast ins(%2 : tensor<64xf32>) outs(%0 : tensor<64x64x64xf32>) dimensions = [1,2] {tensor_ext.layout = [[#layout]]}
      %broadcasted = linalg.broadcast ins(%2 : tensor<64xf32>) outs(%0 : tensor<64x64x64xf32>) dimensions = [1,2]
      secret.yield %broadcasted: tensor<64x64x64xf32>
    } -> !secret.secret<tensor<64x64x64xf32>>
  return %1 : !secret.secret<tensor<64x64x64xf32>>
  }
}
