// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics=ciphertext-size=4096 --mlir-elide-elementsattrs-if-larger=10 | tee /tmp/heir_debug.mlir | FileCheck %s
// Tests a sample from a layernorm of 4 tokens.

// CHECK: func.func @main
// CHECK-NOT: linalg.broadcast
// CHECK-DAG: %[[cst:.*]] = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
// CHECK-DAG: %[[c_256:.*]] = arith.constant -256 : index
// CHECK-DAG: %[[c_128:.*]] = arith.constant -128 : index
// CHECK-DAG: %[[c_64:.*]] = arith.constant -64 : index
// CHECK-DAG: %[[c_32:.*]] = arith.constant -32 : index
// CHECK-DAG: %[[c_16:.*]] = arith.constant -16 : index
// CHECK-DAG: %[[c_8:.*]] = arith.constant -8 : index
// CHECK-DAG: %[[c_4:.*]] = arith.constant -4 : index
// CHECK-DAG: %[[cst_0:.*]] = arith.constant dense_resource<__elided__> : tensor<1x4096xf32>
// CHECK-DAG: %[[ASSIGN:.*]] = tensor.empty()
// CHECK-DAG: arith.mulf %{{.*}}, %[[cst_0]]
// CHECK: tensor_ext.rotate %{{.*}}, %[[c_4]]
// CHECK: tensor_ext.rotate %{{.*}}, %[[c_8]]
// CHECK: tensor_ext.rotate %{{.*}}, %[[c_16]]
// CHECK: tensor_ext.rotate %{{.*}}, %[[c_32]]
// CHECK: tensor_ext.rotate %{{.*}}, %[[c_64]]
// CHECK: tensor_ext.rotate %{{.*}}, %[[c_128]]
// CHECK: tensor_ext.rotate %{{.*}}, %[[c_256]]
// CHECK: arith.mulf %{{.*}}, %[[cst]]
// CHECK: arith.addf %{{.*}}, %[[ASSIGN]]



#layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-768i0 - i1 + slot) mod 4096 = 0 and 0 <= i0 <= 3 and 0 <= i1 <= 4095 - 768i0 and i1 <= 767 and 0 <= slot <= 4095 }">
#layout1 = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-768i0 + slot) mod 4096 = 0 and 0 <= i0 <= 3 and 0 <= slot <= 4095 }">
module {
  func.func @main(%arg0: !secret.secret<tensor<4x768xf32>> {tensor_ext.layout = #layout}) -> (!secret.secret<tensor<4x768xf32>> {tensor_ext.layout = #layout}) {
    %cst = arith.constant dense<7.680000e-02> : tensor<4xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<4xf32>
    %0 = tensor.empty() : tensor<4x768xf32>
    %1 = secret.generic(%arg0: !secret.secret<tensor<4x768xf32>> {tensor_ext.layout = #layout}) {
    ^body(%input0: tensor<4x768xf32>):
      %2 = tensor_ext.assign_layout %cst_0 {layout = #layout1, tensor_ext.layout = #layout1} : tensor<4xf32>
      %reduced = linalg.reduce ins(%input0 : tensor<4x768xf32>) outs(%2 : tensor<4xf32>) dimensions = [1]  {tensor_ext.layout = #layout1}
        (%in: f32, %init: f32) {
          %8 = arith.addf %in, %init : f32
          linalg.yield %8 : f32
        }
      %3 = tensor_ext.assign_layout %cst {layout = #layout1, tensor_ext.layout = #layout1} : tensor<4xf32>
      %4 = arith.mulf %reduced, %3 {tensor_ext.layout = #layout1} : tensor<4xf32>
      %5 = tensor_ext.assign_layout %0 {layout = #layout, tensor_ext.layout = #layout} : tensor<4x768xf32>
      %broadcasted = linalg.broadcast ins(%4 : tensor<4xf32>) outs(%5 : tensor<4x768xf32>) dimensions = [1]  {tensor_ext.layout = #layout}
      %6 = arith.subf %input0, %broadcasted {tensor_ext.layout = #layout} : tensor<4x768xf32>
      %7 = arith.mulf %6, %6 {tensor_ext.layout = #layout} : tensor<4x768xf32>
      secret.yield %7 : tensor<4x768xf32>
    } -> (!secret.secret<tensor<4x768xf32>> {tensor_ext.layout = #layout})
    return %1 : !secret.secret<tensor<4x768xf32>>
  }
}
