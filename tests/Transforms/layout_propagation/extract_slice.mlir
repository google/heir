// RUN: heir-opt --layout-propagation=ciphertext-size=16 --split-input-file %s | FileCheck %s

// CHECK: #[[layout:.*]] = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-4i0 - i1 + slot) mod 32 = 0 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= slot <= 15 }">

module {
  // CHECK: func.func @main
  // CHECK: tensor.extract_slice
  // CHECK-SAME: {tensor_ext.layout = #[[layout]]}
  // CHECK: return
  func.func @main(%arg0: !secret.secret<tensor<1x2x4x4xf32>>) -> !secret.secret<tensor<4x4xf32>> {
    %2 = secret.generic(%arg0: !secret.secret<tensor<1x2x4x4xf32>>) {
    ^body(%input0: tensor<1x2x4x4xf32>):
      %extracted_slice = tensor.extract_slice %input0[0, 0, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : tensor<1x2x4x4xf32> to tensor<4x4xf32>
      secret.yield %extracted_slice : tensor<4x4xf32>
    } -> !secret.secret<tensor<4x4xf32>>
    return %2 : !secret.secret<tensor<4x4xf32>>
  }
}

// -----

// CHECK: #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-4i0 - i1 + slot) mod 32 = 0 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= slot <= 15 }">

module {
  // CHECK: func.func @offset
  // CHECK: tensor.extract_slice
  // CHECK-SAME: {tensor_ext.layout = #[[layout]]}
  // CHECK: return
  func.func @offset(%arg0: !secret.secret<tensor<2x1x4x4xf32>>) -> !secret.secret<tensor<4x4xf32>> {
    %2 = secret.generic(%arg0: !secret.secret<tensor<2x1x4x4xf32>>) {
    ^body(%input0: tensor<2x1x4x4xf32>):
      %extracted_slice = tensor.extract_slice %input0[1, 0, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : tensor<2x1x4x4xf32> to tensor<4x4xf32>
      secret.yield %extracted_slice : tensor<4x4xf32>
    } -> !secret.secret<tensor<4x4xf32>>
    return %2 : !secret.secret<tensor<4x4xf32>>
  }
}
