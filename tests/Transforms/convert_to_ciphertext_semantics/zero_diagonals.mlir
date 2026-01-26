// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics | FileCheck %s

#kernel = #secret.kernel<name = "MatvecDiagonal", force = false>
#layout2 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-32i0 - i1 + slot) mod 1024 = 0 and 0 <= i0 <= 31 and 0 <= i1 <= 31 and 0 <= slot <= 1023 }">
#layout3 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (-32i0 - i1 + ct - 4*floor((slot)/28)) mod 1024 = 0 and 0 <= i0 <= 4 and 0 <= i1 <= 4 and 0 <= ct <= 1023 and slot >= 0 and -28i0 <= slot <= 895 - 28i0 and slot <= 783 and -32i0 - i1 - slot <= 4*floor((slot)/28) <= 1023 - 32i0 - i1 - slot and 28*floor((slot)/28) >= -31 + i1 + slot and 28*floor((slot)/28) <= i1 + slot }">
#layout4 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-28i0 - i1 + slot) mod 1024 = 0 and 0 <= i0 <= 27 and 0 <= i1 <= 1023 - 28i0 and i1 <= 27 and 0 <= slot <= 1023 }">
module {
  // CHECK: func.func @conv2d
  func.func @conv2d(%arg0: !secret.secret<tensor<32x32xf32>> {tensor_ext.layout = #layout2}) -> (!secret.secret<tensor<28x28xf32>> {tensor_ext.layout = #layout4}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<28x28xf32>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<5x5xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<32x32xf32>> {tensor_ext.layout = #layout2}) {
    ^body(%input0: tensor<32x32xf32>):
    // CHECK: secret.generic
    // The plaintext matrix has size 1024x1024 but only 241 rows are non-zero.
    // CHECK: func.call @_assign_layout_{{[0-9]+}}
    // CHECK-SAME: tensor<1024x1024xf32>
    // CHECK: func.call @_assign_layout_{{[0-9]+}}
    // CHECK-SAME: tensor<1x1024xf32>
    // CHECK-COUNT-241: arith.mulf
    // CHECK-NOT: arith.mulf
    // CHECK: secret.yield
      %7 = tensor_ext.assign_layout %cst_0 {layout = #layout3, tensor_ext.layout = #layout3} : tensor<5x5xf32>
      %8 = tensor_ext.assign_layout %cst {layout = #layout4, tensor_ext.layout = #layout4} : tensor<28x28xf32>
      %9 = linalg.conv_2d {secret.kernel = #kernel, tensor_ext.layout = #layout4} ins(%input0, %7 : tensor<32x32xf32>, tensor<5x5xf32>) outs(%8 : tensor<28x28xf32>) -> tensor<28x28xf32>
      secret.yield %9 : tensor<28x28xf32>
    } -> (!secret.secret<tensor<28x28xf32>> {tensor_ext.layout = #layout4})
    return %0 : !secret.secret<tensor<28x28xf32>>
  }
}
