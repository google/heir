// RUN: heir-opt %s --convert-to-ciphertext-semantics=ciphertext-size=4096 --split-input-file | FileCheck %s

// This is a regression test for a bug found - if layout materialization attrs
// aren't added to the layout assignment op, then
// convert-to-ciphertext-semantics hangs indefinitely with no output. This
// ensures that the attrs are added correctly and the pass terminates
// succecssfully.

#layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = 0 and (-196i1 - 14i2 - i3 + slot) mod 1024 = 0 and 0 <= i1 <= 3 and 0 <= i2 <= 13 and 0 <= i3 <= 4095 - 196i1 - 14i2 and i3 <= 13 and 0 <= slot <= 4095 and 4096*floor((-1024 + 196i1 + 14i2 + i3)/4096) <= -4096 + 196i1 + 14i2 + i3 }">
module {
  // CHECK: func.func @complicated_layout
  // CHECK: arith.constant dense<0.000000e+00> : tensor<1x4096xf32>
  func.func @complicated_layout_dense() -> (!secret.secret<tensor<1x4x14x14xf32>> {tensor_ext.layout = #layout}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x4x14x14xf32>
    %0 = secret.generic() {
      %2 = tensor_ext.assign_layout %cst {layout = #layout, tensor_ext.layout = #layout} : tensor<1x4x14x14xf32>
      secret.yield %2 : tensor<1x4x14x14xf32>
    } -> (!secret.secret<tensor<1x4x14x14xf32>> {tensor_ext.layout = #layout})
    return %0 : !secret.secret<tensor<1x4x14x14xf32>>
  }
}


// -----

#layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = 0 and (-196i1 - 14i2 - i3 + slot) mod 1024 = 0 and 0 <= i1 <= 3 and 0 <= i2 <= 13 and 0 <= i3 <= 4095 - 196i1 - 14i2 and i3 <= 13 and 0 <= slot <= 4095 and 4096*floor((-1024 + 196i1 + 14i2 + i3)/4096) <= -4096 + 196i1 + 14i2 + i3 }">
module {
  // CHECK: func.func private @_assign_layout
  // CHECK: func.func @complicated_layout
  func.func @complicated_layout_dense() -> (!secret.secret<tensor<1x4x14x14xf32>> {tensor_ext.layout = #layout}) {
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1x4x14x14xf32>
    %cst_0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
    %cst = linalg.broadcast ins(%cst_0 : tensor<4xf32>) outs(%cst_1 : tensor<1x4x14x14xf32>) dimensions = [0, 2, 3]
    %0 = secret.generic() {
      %2 = tensor_ext.assign_layout %cst {layout = #layout, tensor_ext.layout = #layout} : tensor<1x4x14x14xf32>
      secret.yield %2 : tensor<1x4x14x14xf32>
    } -> (!secret.secret<tensor<1x4x14x14xf32>> {tensor_ext.layout = #layout})
    return %0 : !secret.secret<tensor<1x4x14x14xf32>>
  }
}
