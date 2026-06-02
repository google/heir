// RUN: heir-opt %s --convert-to-ciphertext-semantics | FileCheck %s

#layout_in = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and i2 = 0 and ct = 0 and (-i1 + slot) mod 4 = 0 and 0 <= i1 <= 2 and 0 <= slot <= 1023 }">
#layout_out = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 4 = 0 and 0 <= i0 <= 2 and 0 <= slot <= 1023 }">
module {
  // CHECK: func.func @collapse_multiple_unit_dims
  func.func @collapse_multiple_unit_dims(%arg0: !secret.secret<tensor<1x3x1xf32>> {tensor_ext.layout = #layout_in}) -> (!secret.secret<tensor<3xf32>> {tensor_ext.layout = #layout_out}) {
    // CHECK: secret.generic
    // CHECK-NEXT: ^body(%[[input0:.*]]: tensor<1x1024xf32>)
    // CHECK: secret.yield %[[input0]]
    %0 = secret.generic(%arg0: !secret.secret<tensor<1x3x1xf32>> {tensor_ext.layout = #layout_in}) {
    ^body(%input0: tensor<1x3x1xf32>):
      %collapsed = tensor.collapse_shape %input0 [[0, 1, 2]] {tensor_ext.layout = #layout_out} : tensor<1x3x1xf32> into tensor<3xf32>
      secret.yield %collapsed : tensor<3xf32>
    } -> (!secret.secret<tensor<3xf32>> {tensor_ext.layout = #layout_out})
    return %0 : !secret.secret<tensor<3xf32>>
  }
}
