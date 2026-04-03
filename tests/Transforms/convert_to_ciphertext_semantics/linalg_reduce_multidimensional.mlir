// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics=ciphertext-size=1024 --verify-diagnostics
// Test that a reduction over 2 dimensions fails
#layout = #tensor_ext.layout<"{ [] -> [ct, slot] : ct = 0 and 0 <= slot <= 1023 }">
module {
  func.func @main(%arg0: !secret.secret<tensor<8x8xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-8i0 - i1 + slot) mod 64 = 0 and 0 <= i0 <= 7 and 0 <= i1 <= 7 and 0 <= slot <= 1023 }">}, %arg1: !secret.secret<tensor<8x8xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-8i0 - i1 + slot) mod 64 = 0 and 0 <= i0 <= 7 and 0 <= i1 <= 7 and 0 <= slot <= 1023 }">}) -> (!secret.secret<tensor<f32>> {tensor_ext.layout = #layout}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %0 = tensor.empty() : tensor<8x8xf32>
    // expected-error @+1 {{linalg.reduce only supported with a single reduction dimension}}
    %reduced = linalg.reduce ins(%0 : tensor<8x8xf32>) outs(%cst : tensor<f32>) dimensions = [0, 1]
      (%in: f32, %init: f32) {
        %3 = arith.addf %in, %init : f32
        linalg.yield %3 : f32
      }
    %1 = secret.conceal %reduced : tensor<f32> -> !secret.secret<tensor<f32>>
    %2 = tensor_ext.assign_layout %1 {layout = #layout, tensor_ext.layout = #layout} : !secret.secret<tensor<f32>>
    // expected-error @+1 {{unexpected unrealized conversion cast op found}}
    return %2 : !secret.secret<tensor<f32>>
  }
}
