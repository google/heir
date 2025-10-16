// RUN: heir-opt %s --convert-to-ciphertext-semantics=ciphertext-size=4 | FileCheck %s

// A 2x2 matvec with a plaintext matrix is lowered via the Halevi-Shoup
// diagonal packing/kernel.

#kernel = #secret.kernel<name = "MatvecDiagonal", force = false>
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 4 = 0 and 0 <= i0 < 4 and 0 <= slot < 4 }">
#layout1 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (i0 - i1 + ct) mod 4 = 0 and (-i0 + slot) mod 4 = 0 and 0 <= i0 < 4 and 0 <= i1 < 4 and 0 <= ct < 4 and 0 <= slot < 4 }">

// CHECK: @matvec
func.func @matvec(
    %arg0: !secret.secret<tensor<4xf32>> {tensor_ext.layout = #layout}
) -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #layout}) {
  %cst = arith.constant dense<0.000000e+00> : tensor<4xf32>
  %cst_0 = arith.constant dense<2.0> : tensor<4x4xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<4xf32>> {tensor_ext.layout = #layout}) {
  ^body(%input0: tensor<4xf32>):
    %1 = tensor_ext.assign_layout %cst_0 {layout = #layout1, tensor_ext.layout = #layout1} : tensor<4x4xf32>
    %2 = tensor_ext.assign_layout %cst {layout = #layout, tensor_ext.layout = #layout} : tensor<4xf32>
    %3 = linalg.matvec {secret.kernel = #kernel, tensor_ext.layout = #layout}
          ins(%1, %input0 : tensor<4x4xf32>, tensor<4xf32>)
          outs(%2 : tensor<4xf32>) -> tensor<4xf32>
    secret.yield %3 : tensor<4xf32>
  } -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #layout})
  return %0 : !secret.secret<tensor<4xf32>>
}
