// RUN: heir-opt %s --convert-to-ciphertext-semantics --split-input-file | FileCheck %s

#layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 16 = 0 and 0 <= i1 <= 9 and 0 <= slot <= 1023 }">
#layout4 = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 16 = 0 and 0 <= i0 <= 9 and 0 <= slot <= 1023 }">
module {
  // CHECK: func.func @main
  func.func @main(%arg0: !secret.secret<tensor<10xf32>> {tensor_ext.layout = #layout4}) -> (!secret.secret<tensor<1x10xf32>> {tensor_ext.layout = #layout}) {
    // CHECK: secret.generic
    // CHECK: ^body(%[[v12:.*]]: tensor<1x1024xf32>)
    // CHECK: secret.yield %[[v12]]
    %7 = secret.generic(%arg0: !secret.secret<tensor<10xf32>> {tensor_ext.layout = #layout4}) {
    ^body(%12: tensor<10xf32>):
      %expanded = tensor.expand_shape %12 [[0, 1]] output_shape [1, 10] {tensor_ext.layout = #layout} : tensor<10xf32> into tensor<1x10xf32>
      secret.yield %expanded : tensor<1x10xf32>
    } -> (!secret.secret<tensor<1x10xf32>> {tensor_ext.layout = #layout})
    return %7 : !secret.secret<tensor<1x10xf32>>
  }
}
