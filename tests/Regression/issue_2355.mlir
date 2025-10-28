// RUN: heir-opt --implement-shift-network --canonicalize %s | FileCheck %s

#layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = i1 and (2i2 - i3 + slot) mod 4 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 1 and 0 <= i3 <= 1 and 0 <= slot <= 15 }">
#layout1 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : exists (e0, e1, e2: i0 = 0 and ct = 1 and 4e1 = -i1 + slot + 4e0 and 0 <= i1 <= 15 and 0 <= slot <= 15 and 0 <= e2 <= 1 and -i1 + 4e0 <= 2e2 <= 1 - i1 + 4e0) }">
#original_type = #tensor_ext.original_type<originalType = tensor<1x2x2x2xf32>, layout = #layout>
module {
  func.func @main(%arg0: tensor<1x16xf32> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<1x1x4x4xf32>, layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and i1 = 0 and ct = 0 and (-4i2 - i3 + slot) mod 16 = 0 and 0 <= i2 <= 3 and 0 <= i3 <= 3 and 0 <= slot <= 15 }">>}, %arg1: tensor<2x1x3x3xf32>, %arg2: tensor<1x16xf32>, %arg3: tensor<2x16xf32>) -> (tensor<2x16xf32> {tensor_ext.original_type = #original_type}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x16xf32>
    %0 = tensor.empty() : tensor<2x16xf32>
    %1 = tensor_ext.remap %arg2 {permutation = #layout1} : tensor<1x16xf32>
    %extracted_slice = tensor.extract_slice %arg3[0, 0] [1, 16] [1, 1] : tensor<2x16xf32> to tensor<1x16xf32>
    %2 = arith.addf %extracted_slice, %cst fastmath<nnan,nsz> : tensor<1x16xf32>
    %inserted_slice = tensor.insert_slice %2 into %0[0, 0] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<2x16xf32>
    %3 = arith.addf %1, %cst fastmath<nnan,nsz> : tensor<1x16xf32>
    %inserted_slice_0 = tensor.insert_slice %3 into %inserted_slice[1, 0] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<2x16xf32>
    return %inserted_slice_0 : tensor<2x16xf32>
  }
}
