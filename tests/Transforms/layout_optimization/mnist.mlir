// RUN: heir-opt --layout-optimization %s | FileCheck %s

#alignment = #tensor_ext.alignment<in = [1, 10], out = [1, 1024], padding = [0, 6], paddingValue = 0.000000e+00 : f32>
#alignment1 = #tensor_ext.alignment<in = [1, 784], out = [1, 1024], padding = [0, 240], paddingValue = 0.000000e+00 : f32>
#alignment2 = #tensor_ext.alignment<in = [784], out = [1024], padding = [240], paddingValue = 0.000000e+00 : f32>
#alignment3 = #tensor_ext.alignment<in = [784, 512], out = [1024, 512], padding = [240, 0], paddingValue = 0.000000e+00 : f32>
#alignment4 = #tensor_ext.alignment<in = [512], out = [1024]>
#alignment5 = #tensor_ext.alignment<in = [512, 10], out = [512, 16], padding = [0, 6], paddingValue = 0.000000e+00 : f32>
#alignment6 = #tensor_ext.alignment<in = [10], out = [1024], padding = [6], paddingValue = 0.000000e+00 : f32>
#layout = #tensor_ext.layout<map = (d0, d1) -> (d1 mod 1024), alignment = #alignment>
#layout1 = #tensor_ext.layout<map = (d0, d1) -> (d1 mod 1024), alignment = #alignment1>
#layout2 = #tensor_ext.layout<map = (d0) -> (d0 mod 1024), alignment = #alignment2>
#layout3 = #tensor_ext.layout<map = (d0, d1) -> (((d0 * 512 + d1) floordiv 1024) mod 512, (d0 * 512 + d1) mod 1024), alignment = #alignment3>
#layout4 = #tensor_ext.layout<map = (d0) -> (d0 mod 1024), alignment = #alignment4>
#layout5 = #tensor_ext.layout<map = (d0, d1) -> (((d0 * 16 + d1) floordiv 1024) mod 8, (d0 * 16 + d1) mod 1024), alignment = #alignment5>
#layout6 = #tensor_ext.layout<map = (d0) -> (d0 mod 1024), alignment = #alignment6>
module attributes {backend.openfhe, scheme.bgv} {

  // CHECK: @main
  func.func @main(%arg0: tensor<512x784xf32> {debug.name = "layer1_weights"}, %arg1: tensor<512xf32> {debug.name = "layer1_bias"}, %arg2: tensor<10x512xf32> {debug.name = "layer2_weights"}, %arg3: tensor<10xf32> {debug.name = "layer2_bias"}, %arg4: !secret.secret<tensor<1x784xf32>> {debug.name = "input", tensor_ext.layout = #layout1}) -> (!secret.secret<tensor<1x10xf32>> {debug.name = "result", tensor_ext.layout = #layout}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<10xf32>
    %0 = tensor.empty() : tensor<784x512xf32>
    %1 = tensor.empty() : tensor<512x10xf32>
    %2 = secret.generic(%arg4: !secret.secret<tensor<1x784xf32>> {tensor_ext.layout = #layout1}) {
    ^body(%input0: tensor<1x784xf32>):
      %transposed = linalg.transpose ins(%arg0 : tensor<512x784xf32>) outs(%0 : tensor<784x512xf32>) permutation = [1, 0]
      %collapsed = tensor.collapse_shape %input0 [[0, 1]] {tensor_ext.layout = #layout2} : tensor<1x784xf32> into tensor<784xf32>
      %3 = tensor_ext.assign_layout %transposed {layout = #layout3, tensor_ext.layout = #layout3} : tensor<784x512xf32>
      %4 = tensor_ext.assign_layout %cst {layout = #layout4, tensor_ext.layout = #layout4} : tensor<512xf32>
      %5 = linalg.vecmat {tensor_ext.layout = #layout2} ins(%collapsed, %3 : tensor<784xf32>, tensor<784x512xf32>) outs(%4 : tensor<512xf32>) -> tensor<512xf32>
      %6 = tensor_ext.assign_layout %arg1 {layout = #layout4, tensor_ext.layout = #layout4} : tensor<512xf32>
      %7 = tensor_ext.convert_layout %5 {from_layout = #layout2, tensor_ext.layout = #layout4, to_layout = #layout4} : tensor<512xf32>
      %8 = arith.addf %6, %7 {tensor_ext.layout = #layout4} : tensor<512xf32>
      %9 = tensor_ext.assign_layout %cst {layout = #layout4, tensor_ext.layout = #layout4} : tensor<512xf32>
      %10 = arith.maximumf %8, %9 {tensor_ext.layout = #layout4} : tensor<512xf32>
      %transposed_1 = linalg.transpose ins(%arg2 : tensor<10x512xf32>) outs(%1 : tensor<512x10xf32>) permutation = [1, 0]
      %11 = tensor_ext.assign_layout %transposed_1 {layout = #layout5, tensor_ext.layout = #layout5} : tensor<512x10xf32>
      %12 = tensor_ext.assign_layout %cst_0 {layout = #layout6, tensor_ext.layout = #layout6} : tensor<10xf32>
      %13 = linalg.vecmat {tensor_ext.layout = #layout4} ins(%10, %11 : tensor<512xf32>, tensor<512x10xf32>) outs(%12 : tensor<10xf32>) -> tensor<10xf32>
      %14 = tensor_ext.assign_layout %arg3 {layout = #layout6, tensor_ext.layout = #layout6} : tensor<10xf32>
      %15 = tensor_ext.convert_layout %13 {from_layout = #layout4, tensor_ext.layout = #layout6, to_layout = #layout6} : tensor<10xf32>
      %16 = arith.addf %14, %15 {tensor_ext.layout = #layout6} : tensor<10xf32>
      %expanded = tensor.expand_shape %16 [[0, 1]] output_shape [1, 10] {tensor_ext.layout = #layout} : tensor<10xf32> into tensor<1x10xf32>
      secret.yield %expanded : tensor<1x10xf32>
    } -> (!secret.secret<tensor<1x10xf32>> {tensor_ext.layout = #layout})
    return %2 : !secret.secret<tensor<1x10xf32>>
  }
}
