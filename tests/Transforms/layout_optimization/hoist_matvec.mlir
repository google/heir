// RUN: heir-opt --layout-optimization %s | FileCheck %s

#vec_align = #tensor_ext.alignment<in = [512], out = [512]>
#vec_layout = #tensor_ext.layout<map = (d0) -> (d0 + 4 mod 512), alignment = #vec_align>
#vec_layout_2 = #tensor_ext.layout<map = (d0) -> (d0 + 7 mod 512), alignment = #vec_align>

#mat_align = #tensor_ext.alignment<in = [512, 512], out = [512, 512]>
#mat_layout = #tensor_ext.layout<map = (d0, d1) -> (((d0 * 512 + d1) floordiv 512) mod 512, (d0 * 512 + d1) mod 512), alignment = #mat_align>

// CHECK: #tensor_ext.layout<map = (d0, d1) -> (((d0 * 512 + d1) floordiv 512) mod 512, ((d0 * 512 + d1) mod 512) + 7 mod 512), alignment = #mat_align>
// In both cases: precompose the transformation of the output vec layout with the d1, so long as invertible

d0 -> 0
d0 -> d0 floordiv 2

a0 0 a1 0 a2 0 a3 0
a0 a0 a1 a1 a2 a2

d0 -> 2*d0

module attributes {
  func.func @main(
      %arg0: tensor<512x512xf32>,
      %input: !secret.secret<tensor<512xf32>> {tensor_ext.layout = #vec_align}
    ) -> (!secret.secret<tensor<512xf32>> {tensor_ext.layut = #vec_layout_2}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %1 = tensor.empty() : tensor<512xf32>
    %4 = tensor_ext.assign_layout %1 {layout = #vec_align, tensor_ext.layout = #vec_align} : tensor<512xf32>
    %6 = tensor_ext.assign_layout %arg0 {layout = #mat_align, tensor_ext.layout = #mat_align} : tensor<512x512xf32>
    %2 = secret.generic(%input: !secret.secret<tensor<512xf32>> {tensor_ext.layout = #vec_align}) {
    ^body(%input0: tensor<512xf32>):
      %5 = linalg.matvec {tensor_ext.layout = #vec_align} ins(%6, %input0 : tensor<512x512xf32>, tensor<512xf32>) outs(%4 : tensor<512xf32>) -> tensor<512xf32>
      %6 = tensor_ext.convert_layout {from_layout = #vec_layout, to_layout = #vec_layout_2, tensor_ext.layout = #vec_layout_2} : tensor<512xf32>
      secret.yield %6 : tensor<512xf32>
    } -> (!secret.secret<tensor<512xf32>> {tensor_ext.layout = #vec_layout_2})
    return %2 : !secret.secret<tensor<512xf32>>
  }
}
