// RUN: heir-opt %s --convert-to-ciphertext-semantics=min-slot-count=1024 | FileCheck %s

// A reduction over two dimensions has no dedicated kernel, so the generic
// kernel implements it. The input repeats every 64 slots and the result is
// replicated in every slot, so each of the 64 distinct rotations is one
// all-ones group.

#in = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-8i0 - i1 + slot) mod 64 = 0 and 0 <= i0 <= 7 and 0 <= i1 <= 7 and 0 <= slot <= 1023 }">
#out = #tensor_ext.layout<"{ [] -> [ct, slot] : ct = 0 and 0 <= slot <= 1023 }">

// CHECK: func.func @sum_all
// CHECK-NOT: linalg.reduce
// CHECK-COUNT-63: tensor_ext.rotate
// CHECK-NOT: tensor_ext.rotate
// CHECK-NOT: arith.mulf
// CHECK: return
func.func @sum_all(%arg0: !secret.secret<tensor<8x8xf32>> {tensor_ext.layout = #in}) -> (!secret.secret<tensor<f32>> {tensor_ext.layout = #out}) {
  %zero = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<8x8xf32>> {tensor_ext.layout = #in}) {
  ^body(%input: tensor<8x8xf32>):
    %init = tensor_ext.assign_layout %zero <layout = #out> {tensor_ext.layout = #out} : tensor<f32>
    %sum = linalg.reduce ins(%input : tensor<8x8xf32>) outs(%init : tensor<f32>) dimensions = [0, 1] {tensor_ext.layout = #out}
      (%in: f32, %acc: f32) {
        %1 = arith.addf %in, %acc : f32
        linalg.yield %1 : f32
      }
    secret.yield %sum : tensor<f32>
  } -> (!secret.secret<tensor<f32>> {tensor_ext.layout = #out})
  return %0 : !secret.secret<tensor<f32>>
}
