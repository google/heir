// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics=ciphertext-size=1024 | FileCheck %s
// Tests that a broadcast for the 0th dimension gets converted and also broadcasting non power of 2 dimensions.

// CHECK: func.func @main
// CHECK-NOT: linalg.broadcast
// CHECK-DAG: %[[c_90:.*]] = arith.constant -90 : index
// CHECK-DAG: %[[c_180:.*]] = arith.constant -180 : index
// CHECK-DAG: %[[c_270:.*]] = arith.constant -270 : index
// CHECK-DAG: %[[cst:.*]] = arith.constant dense<
// CHECK-DAG: %[[cst_0:.*]] = arith.constant dense<"0x0000803F0000803F
// CHECK-DAG: %[[ASSIGN:.*]] = tensor.empty()
// CHECK: tensor_ext.rotate %{{.*}}, %[[c_90]]
// CHECK: tensor_ext.rotate %{{.*}}, %[[c_180]]
// CHECK: tensor_ext.rotate %{{.*}}, %[[c_270]]
// CHECK: arith.addf %{{.*}}, %[[ASSIGN]]
#layout = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : ct = 0 and (-90i0 - 9i1 - i2 + slot) mod 1024 = 0 and 0 <= i0 <= 6 and 0 <= i1 <= 9 and 0 <= i2 <= 1023 - 90i0 - 9i1 and i2 <= 8 and 0 <= slot <= 1023 }">
#layout1 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-9i0 - i1 + slot) mod 128 = 0 and 0 <= i0 <= 9 and 0 <= i1 <= 1023 - 9i0 and i1 <= 8 and 0 <= slot <= 1023 and 1024*floor((-128 + 9i0 + i1)/1024) <= -1024 + 9i0 + i1 }">
module {
  func.func @main(%arg0: !secret.secret<tensor<10x9xf32>> {secret.secret, tensor_ext.layout = #layout1}) -> (!secret.secret<tensor<7x10x9xf32>> {secret.secret, tensor_ext.layout = #layout}) {
    %0 = tensor.empty() {secret.public} : tensor<7x10x9xf32>
    %1 = secret.generic(%arg0: !secret.secret<tensor<10x9xf32>> {tensor_ext.layout = #layout1}) attrs = {secret.secret} {
    ^body(%input0: tensor<10x9xf32>):
      %2 = tensor_ext.assign_layout %0 {layout = #layout, tensor_ext.layout = #layout} : tensor<7x10x9xf32>
      %broadcasted = linalg.broadcast ins(%input0 : tensor<10x9xf32>) outs(%2 : tensor<7x10x9xf32>) dimensions = [0]  {secret.secret, tensor_ext.layout = #layout}
      secret.yield %broadcasted {secret.secret} : tensor<7x10x9xf32>
    } -> (!secret.secret<tensor<7x10x9xf32>> {tensor_ext.layout = #layout})
    return {secret.secret} %1 : !secret.secret<tensor<7x10x9xf32>>
  }
}
