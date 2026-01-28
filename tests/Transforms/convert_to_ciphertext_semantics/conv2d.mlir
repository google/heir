// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics=ciphertext-size=32 | FileCheck %s

#kernel = #secret.kernel<name = "MatvecDiagonal", force = false>
#layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-3i0 - i1 + slot) mod 16 = 0 and 0 <= i0 <= 2 and 0 <= i1 <= 31 - 3i0 and i1 <= 2 and 0 <= slot <= 31 and 32*floor((16 + 3i0 + i1)/32) <= 3i0 + i1 }">
#layout1 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-5i0 - i1 + slot) mod 32 = 0 and 0 <= i0 <= 4 and 0 <= i1 <= 31 - 5i0 and i1 <= 4 and 0 <= slot <= 31 }">
#layout2 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (-3i0 - i1 + ct + 8*floor((slot)/8)) mod 16 = 0 and 0 <= i0 <= 2 and 0 <= i1 <= 2 and 0 <= ct <= 7 and 0 <= slot <= 31 and 8*floor((slot)/8) >= -15 + 3i0 + i1 + slot and 8*floor((slot)/8) >= -14 + 3i0 + slot and 8*floor((slot)/8) <= 3i0 + slot and 8*floor((slot)/8) <= 3i0 + i1 + slot and -39 + 8i1 + 9slot <= 24*floor((3slot)/8) <= 8i1 + 9slot }">
module {
  // CHECK: func.func @conv2d
  func.func @conv2d(%arg0: !secret.secret<tensor<5x5xf32>> {tensor_ext.layout = #layout1}) -> (!secret.secret<tensor<3x3xf32>> {tensor_ext.layout = #layout}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<3x3xf32>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<3x3xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<5x5xf32>> {tensor_ext.layout = #layout1}) {
    ^body(%input0: tensor<5x5xf32>):
    // CHECK: secret.generic
    // CHECK: func.call @_assign_layout_{{[0-9]+}}
    // CHECK: func.call @_assign_layout_{{[0-9]+}}
    // CHECK-COUNT-13: tensor_ext.rotate
      %1 = tensor_ext.assign_layout %cst_0 {layout = #layout2, tensor_ext.layout = #layout2} : tensor<3x3xf32>
      %2 = tensor_ext.assign_layout %cst {layout = #layout, tensor_ext.layout = #layout} : tensor<3x3xf32>
      %3 = linalg.conv_2d {secret.kernel = #kernel, tensor_ext.layout = #layout} ins(%input0, %1 : tensor<5x5xf32>, tensor<3x3xf32>) outs(%2 : tensor<3x3xf32>) -> tensor<3x3xf32>
      secret.yield %3 : tensor<3x3xf32>
    } -> (!secret.secret<tensor<3x3xf32>> {tensor_ext.layout = #layout})
    return %0 : !secret.secret<tensor<3x3xf32>>
  }
}
