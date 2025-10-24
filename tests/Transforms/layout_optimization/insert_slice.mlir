// RUN: heir-opt --layout-optimization=ciphertext-size=32 --canonicalize %s | FileCheck %s

// This layout assigned each ciphertext to an individual 4x4 slice.
// CHECK: #[[layout:.*]] = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = i1 and (-4i2 - i3 + slot) mod 16 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 3 and 0 <= i3 <= 3 and 0 <= slot <= 31 }">
#layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = i1 and (-4i2 - i3 + slot) mod 16 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 3 and 0 <= i3 <= 3 and 0 <= slot <= 31 }">
#layout1 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-4i0 - i1 + slot) mod 16 = 0 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= slot <= 31 }">
module {
  // Checks that the layout assigned to the output of tensor.insert_slice is used in the subsequent operations.
  // CHECK: func.func @main
  // CHECK: %[[v0:.*]] = tensor.empty() : tensor<1x2x4x4xf32>
  // CHECK: tensor_ext.assign_layout %[[v0]]
  // CHECK-SAME: {layout = #[[layout]], tensor_ext.layout = #[[layout]]}
  // CHECK: tensor.insert_slice
  // CHECK-SAME: {tensor_ext.layout = #[[layout]]}
  // CHECK: arith.addf
  // CHECK-SAME: {tensor_ext.layout = #[[layout]]}
  // CHECK: return
  func.func @main(%arg0: !secret.secret<tensor<4x4xf32>> {tensor_ext.layout = #layout1}) -> (!secret.secret<tensor<1x2x4x4xf32>> {tensor_ext.layout = #layout}) {
    %0 = tensor.empty() : tensor<1x2x4x4xf32>
    %1 = secret.generic(%arg0: !secret.secret<tensor<4x4xf32>> {tensor_ext.layout = #layout1}) {
    ^body(%input0: tensor<4x4xf32>):
      %2 = tensor_ext.assign_layout %0 {layout = #layout, tensor_ext.layout = #layout} : tensor<1x2x4x4xf32>
      %inserted_slice = tensor.insert_slice %input0 into %2[0, 0, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] {tensor_ext.layout = #layout} : tensor<4x4xf32> into tensor<1x2x4x4xf32>
      %inserted_slice_0 = tensor.insert_slice %input0 into %inserted_slice[0, 1, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] {tensor_ext.layout = #layout} : tensor<4x4xf32> into tensor<1x2x4x4xf32>
      %3 = arith.addf %inserted_slice_0, %inserted_slice_0 {tensor_ext.layout = #layout} : tensor<1x2x4x4xf32>
      secret.yield %3 : tensor<1x2x4x4xf32>
    } -> (!secret.secret<tensor<1x2x4x4xf32>> {tensor_ext.layout = #layout})
    return %1 : !secret.secret<tensor<1x2x4x4xf32>>
  }
}
