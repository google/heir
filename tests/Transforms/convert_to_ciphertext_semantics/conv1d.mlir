// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics=ciphertext-size=32 | FileCheck %s

#kernel = #secret.kernel<name = "MatvecDiagonal", force = false>
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 4 = 0 and 0 <= i0 <= 2 and 0 <= slot <= 1023 }">
#layout1 = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 8 = 0 and 0 <= i0 <= 4 and 0 <= slot <= 1023 }">
#layout2 = #tensor_ext.layout<"{ [i0] -> [ct, slot] : (-i0 + ct + 4*floor((1 + slot)/4)) mod 8 = 0 and 0 <= i0 <= 2 and 0 <= ct <= 3 and 0 <= slot <= 1023 and 4*floor((1 + slot)/4) >= -4 + i0 + slot and 4*floor((1 + slot)/4) <= slot and 4*floor((1 + slot)/4) <= i0 + slot }">
module {
  func.func @conv1d(%arg0: !secret.secret<tensor<5xf32>> {tensor_ext.layout = #layout1}) -> (!secret.secret<tensor<3xf32>> {tensor_ext.layout = #layout}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<3xf32>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<3xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<5xf32>> {tensor_ext.layout = #layout1}) {
    ^body(%input0: tensor<5xf32>):
      %1 = tensor_ext.assign_layout %cst_0 {layout = #layout2, tensor_ext.layout = #layout2} : tensor<3xf32>
      %2 = tensor_ext.assign_layout %cst {layout = #layout1, tensor_ext.layout = #layout1} : tensor<3xf32>
      %3 = linalg.conv_1d {secret.kernel = #kernel, tensor_ext.layout = #layout} ins(%input0, %1 : tensor<5xf32>, tensor<3xf32>) outs(%2 : tensor<3xf32>) -> tensor<3xf32>
      secret.yield %3 : tensor<3xf32>
    } -> (!secret.secret<tensor<3xf32>> {tensor_ext.layout = #layout})
    return %0 : !secret.secret<tensor<3xf32>>
  }
}
