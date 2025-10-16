// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics=ciphertext-size=1024 | FileCheck %s

// Tests a non-default matvec layout can still be lowered to a matvec kernel.

#kernel = #secret.kernel<name = "MatvecDiagonal", force = false>
#layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 32 = 0 and 0 <= i1 <= 19 and 0 <= slot <= 1023 }">
#layout1 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (i0 - i1 + ct) mod 32 = 0 and (-i1 + ct + slot) mod 64 = 0 and 0 <= i0 <= 19 and 0 <= i1 <= 49 and 0 <= ct <= 31 and 0 <= slot <= 1023 }">
#layout2 = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 32 = 0 and 0 <= i0 <= 19 and 0 <= slot <= 1023 }">
#layout3 = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : ct = 0 and (-100i0 - 10i1 - i2 + slot) mod 256 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 4 and 0 <= i2 <= 4 and 0 <= slot <= 1023 and 1024*floor((-256 + 100i0 + 10i1 + i2)/1024) <= -1024 + 100i0 + 10i1 + i2 }">
#layout4 = #tensor_ext.layout<"{ [i0] -> [ct, slot] : exists (e0, e1, e2, e3: ct = 0 and 5e3 = -i0 + slot - 256e1 - 75e2 and 0 <= i0 <= 49 and 0 <= slot <= 1023 and -1279 + slot - 1024e0 <= 256e1 <= -1024 + slot - 1024e0 and 0 <= e2 <= 1 and -2i0 + slot - 256e1 <= 50e2 <= 4 - 2i0 + slot - 256e1 and -20 - i0 + slot - 256e1 <= 75e2 <= -i0 + slot - 256e1) }">
module {
  // CHECK: func.func @main
  // CHECK-NOT: linalg.matvec
  // CHECK: return
  func.func @main(%arg0: !secret.secret<tensor<2x5x5xf32>> {tensor_ext.layout = #layout3}) -> (!secret.secret<tensor<1x20xf32>> {tensor_ext.layout = #layout}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<20xf32>
    %cst_0 = arith.constant dense_resource<torch_tensor_120_400_torch.float32> : tensor<20x50xf32>
    %0 = tensor_ext.assign_layout %cst_0 {layout = #layout1, tensor_ext.layout = #layout1} : tensor<20x50xf32>
    %1 = tensor_ext.assign_layout %cst {layout = #layout2, tensor_ext.layout = #layout2} : tensor<20xf32>
    %2 = secret.generic(%arg0: !secret.secret<tensor<2x5x5xf32>> {tensor_ext.layout = #layout3}) {
    ^body(%input0: tensor<2x5x5xf32>):
      %collapsed = tensor.collapse_shape %input0 [[0, 1, 2]] {tensor_ext.layout = #layout4} : tensor<2x5x5xf32> into tensor<50xf32>
      %3 = linalg.matvec {secret.kernel = #kernel, tensor_ext.layout = #layout2} ins(%0, %collapsed : tensor<20x50xf32>, tensor<50xf32>) outs(%1 : tensor<20xf32>) -> tensor<20xf32>
      %expanded = tensor.expand_shape %3 [[0, 1]] output_shape [1, 20] {tensor_ext.layout = #layout} : tensor<20xf32> into tensor<1x20xf32>
      secret.yield %expanded : tensor<1x20xf32>
    } -> (!secret.secret<tensor<1x20xf32>> {tensor_ext.layout = #layout})
    return %2 : !secret.secret<tensor<1x20xf32>>
  }
}
