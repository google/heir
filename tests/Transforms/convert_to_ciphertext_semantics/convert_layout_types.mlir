// RUN: heir-opt %s --convert-to-ciphertext-semantics | FileCheck %s

// This tests that a tensor_ext.convert_layout operation correctly lowers to a
// remap and extract slice when the layout conversion would result in a tensor
// with a different number of ciphertexts in it. The tensor_ext.convert_layout
// lowering must produce a result that has the correct number of ciphertexts for
// the layout of the output tensor.

// For example, this may occur when squashing data values split across two
// ciphertexts into one single ciphertext, as in the following example.

// The last case converts from a layout given as a list of layouts to compose.

#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 512 = 0 and 0 <= i0 <= 391 and 0 <= slot <= 1023 }">
#layout1 = #tensor_ext.layout<"{ [i0] -> [ct, slot] : (-i0 - 60ct + slot) mod 256 = 0 and 0 <= ct <= 1 and -195 + i0 <= 196ct <= i0 and 0 <= slot <= 1023 }">
#rotate = #tensor_ext.layout<"{ [c, s] -> [ct, slot] : c = 0 and ct = 0 and (slot - s - 1) mod 1024 = 0 and 0 <= s <= 1023 and 0 <= slot <= 1023 }">
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
      %1 = tensor_ext.convert_layout %input0 <from_layout = #layout1, to_layout = #layout> {tensor_ext.layout = #layout} : tensor<392xf32>
      secret.yield %1 : tensor<392xf32>
    } -> (!secret.secret<tensor<392xf32>> {tensor_ext.layout = #layout})
    return %0 : !secret.secret<tensor<392xf32>>
  }

  // Spreading one ciphertext over two.
  // CHECK: func.func @spread
  // CHECK: secret.generic
  // CHECK-NEXT: ^body(%[[input0:.*]]: tensor<1x1024xf32>):
  // CHECK: %[[extended:.*]] = tensor.insert_slice %[[input0]] into %{{.*}}[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> into tensor<2x1024xf32>
  // CHECK: %[[remap:.*]] = tensor_ext.remap %[[extended]] {{.*}} : tensor<2x1024xf32>
  // CHECK: secret.yield %[[remap]] : tensor<2x1024xf32>
  func.func @spread(%arg0: !secret.secret<tensor<392xf32>> {tensor_ext.layout = #layout}) -> (!secret.secret<tensor<392xf32>> {tensor_ext.layout = #layout1}) {
    %0 = secret.generic(%arg0: !secret.secret<tensor<392xf32>> {tensor_ext.layout = #layout}) {
    ^body(%input0: tensor<392xf32>):
      %1 = tensor_ext.convert_layout %input0 <from_layout = #layout, to_layout = #layout1> {tensor_ext.layout = #layout1} : tensor<392xf32>
      secret.yield %1 : tensor<392xf32>
    } -> (!secret.secret<tensor<392xf32>> {tensor_ext.layout = #layout1})
    return %0 : !secret.secret<tensor<392xf32>>
  }

  // A layout recorded as a list of layouts to compose.
  // CHECK: func.func @composed
  // CHECK: secret.generic
  // CHECK-NEXT: ^body(%[[input0:.*]]: tensor<1x1024xf32>):
  // CHECK: %[[remap:.*]] = tensor_ext.remap %[[input0]] {{.*}} : tensor<1x1024xf32>
  // CHECK: secret.yield %[[remap]] : tensor<1x1024xf32>
  func.func @composed(%arg0: !secret.secret<tensor<392xf32>> {tensor_ext.layout = [#layout, #rotate]}) -> (!secret.secret<tensor<392xf32>> {tensor_ext.layout = #layout}) {
    %0 = secret.generic(%arg0: !secret.secret<tensor<392xf32>> {tensor_ext.layout = [#layout, #rotate]}) {
    ^body(%input0: tensor<392xf32>):
      %1 = tensor_ext.convert_layout %input0 <from_layout = [#layout, #rotate], to_layout = #layout> {tensor_ext.layout = #layout} : tensor<392xf32>
      secret.yield %1 : tensor<392xf32>
    } -> (!secret.secret<tensor<392xf32>> {tensor_ext.layout = #layout})
    return %0 : !secret.secret<tensor<392xf32>>
  }
}
