// RUN: heir-opt --convert-to-ciphertext-semantics=ciphertext-size=32 --split-input-file %s | FileCheck %s

#layout1 = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = i1 and (-4i2 - i3 + slot) mod 16 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 3 and 0 <= i3 <= 3 and 0 <= slot <= 31 }">
#layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-4i0 - i1 + slot) mod 16 = 0 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= slot <= 31 }">
module {
  // Layouts are aligned perfectly so that extract_slice extracts a single ciphertext out of %input0
  // CHECK: func.func @trivial_insert
  // CHECK-SAME: (%[[arg0:.*]]: !secret.secret<tensor<2x32xf32>>
  func.func @trivial_insert(%arg0: !secret.secret<tensor<1x2x4x4xf32>> {tensor_ext.layout = #layout1}) -> (!secret.secret<tensor<4x4xf32>> {tensor_ext.layout = #layout}) {
    %1 = secret.generic(%arg0: !secret.secret<tensor<1x2x4x4xf32>> {tensor_ext.layout = #layout1}) {
    ^body(%input0: tensor<1x2x4x4xf32>):
    // CHECK: secret.generic(%[[arg0]]: !secret.secret<tensor<2x32xf32>>)
    // CHECK-NEXT: ^body(%[[input0:.*]]: tensor<2x32xf32>)
    // CHECK: %[[v1:.*]] = tensor_ext.remap %[[input0]]
    // CHECK-NEXT: %[[extracted:.*]] = tensor.extract_slice %[[v1]][0, 0] [1, 32] [1, 1]
    // CHECK-NEXT: %[[v2:.*]] = arith.addf %[[extracted]], %[[extracted]]
    // CHECK-NEXT: secret.yield %[[v2]]
      %extract_slice = tensor.extract_slice %input0 [0, 1, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] {tensor_ext.layout = #layout}
           : tensor<1x2x4x4xf32> to tensor<4x4xf32>
      %3 = arith.addf %extract_slice, %extract_slice {tensor_ext.layout = #layout} : tensor<4x4xf32>
      secret.yield %3 : tensor<4x4xf32>
    } -> (!secret.secret<tensor<4x4xf32>> {tensor_ext.layout = #layout})
    return %1 : !secret.secret<tensor<4x4xf32>>
  }
}
