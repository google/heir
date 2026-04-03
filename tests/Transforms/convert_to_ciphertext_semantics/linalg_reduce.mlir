// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics=ciphertext-size=1024 | FileCheck %s
// Test that a 8 length vector gets reduced.
// CHECK: func.func @main
// CHECK-NOT: linalg.reduce
// CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[ASSIGN:.*]] = arith.constant dense<0{{.*}}> : tensor<1x1024xf32>
// CHECK: tensor_ext.rotate %{{.*}}, %[[c4]]
// CHECK: tensor_ext.rotate %{{.*}}, %[[c2]]
// CHECK: tensor_ext.rotate %{{.*}}, %[[c1]]
// CHECK: arith.addf %{{.*}}, %[[ASSIGN]]
#layout = #tensor_ext.layout<"{ [] -> [ct, slot] : ct = 0 and (slot) mod 8 = 0 and 0 <= slot <= 1023 }">
#layout1 = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 8 = 0 and 0 <= i0 <= 7 and 0 <= slot <= 1023 }">
module {
  func.func @main(%arg0: !secret.secret<tensor<8xf32>> {tensor_ext.layout = #layout1}, %arg1: !secret.secret<tensor<8xf32>> {tensor_ext.layout = #layout1}) -> (!secret.secret<tensor<f32>> {tensor_ext.layout = #layout}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<8xf32>> {tensor_ext.layout = #layout1}, %arg1: !secret.secret<tensor<8xf32>> {tensor_ext.layout = #layout1}) {
^body(%input0: tensor<8xf32>, %input1: tensor<8xf32>):
      %1 = arith.mulf %input0, %input1 {tensor_ext.layout = #layout1} : tensor<8xf32>
      %2 = tensor_ext.assign_layout %cst {layout = #layout, tensor_ext.layout = #layout} : tensor<f32>
      %reduced = linalg.reduce ins(%1 : tensor<8xf32>) outs(%2 : tensor<f32>) dimensions = [0]  {tensor_ext.layout = #layout}
        (%in: f32, %init: f32) {
          %3 = arith.addf %in, %init : f32
          linalg.yield %3 : f32
        }
      secret.yield %reduced : tensor<f32>
    } -> (!secret.secret<tensor<f32>> {tensor_ext.layout = #layout})
    return %0 : !secret.secret<tensor<f32>>
  }
}
