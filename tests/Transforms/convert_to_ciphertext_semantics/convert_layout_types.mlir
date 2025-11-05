// RUN: heir-opt %s --convert-to-ciphertext-semantics | FileCheck %s

// This tests that a tensor_ext.convert_layout operation correctly lowers to a
// remap and extract slice when the layout conversion would result in a tensor
// with a different number of ciphertexts in it. The tensor_ext.convert_layout
// lowering must produce a result that has the correct number of ciphertexts for
// the layout of the output tensor.

// For example, this may occur when squashing data values split across two
// ciphertexts into one single ciphertext, as in the following example.

#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 512 = 0 and 0 <= i0 <= 391 and 0 <= slot <= 1023 }">
#layout1 = #tensor_ext.layout<"{ [i0] -> [ct, slot] : (-i0 - 60ct + slot) mod 256 = 0 and 0 <= ct <= 1 and -195 + i0 <= 196ct <= i0 and 0 <= slot <= 1023 }">
module {
  // CHECK: func.func @main
  // CHECK: secret.generic
  // CHECK-NEXT: ^body(%[[input0:.*]]: tensor<2x1024xf32>):
  // CHECK: %[[remap:.*]] = tensor_ext.remap %[[input0]] {{.*}} : tensor<2x1024xf32>
  // CHECK: %[[extract:.*]] = tensor.extract_slice %[[remap]][0, 0] [1, 1024] [1, 1] : tensor<2x1024xf32> to tensor<1x1024xf32>
  // CHECK: secret.yield %[[extract]] : tensor<1x1024xf32>
  func.func @main(%arg0: !secret.secret<tensor<392xf32>> {tensor_ext.layout = #layout1}) -> (!secret.secret<tensor<392xf32>> {tensor_ext.layout = #layout}) {
    %0 = secret.generic(%arg0: !secret.secret<tensor<392xf32>> {tensor_ext.layout = #layout1}) {
    ^body(%input0: tensor<392xf32>):
      %1 = tensor_ext.convert_layout %input0 {from_layout = #layout1, tensor_ext.layout = #layout, to_layout = #layout} : tensor<392xf32>
      secret.yield %1 : tensor<392xf32>
    } -> (!secret.secret<tensor<392xf32>> {tensor_ext.layout = #layout})
    return %0 : !secret.secret<tensor<392xf32>>
  }
}
