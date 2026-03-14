// RUN: heir-opt --inline-activations %s | FileCheck %s

// This test was extracted from a ResNet-like input, with multiple types of relus.

// CHECK-NOT: call @relu
#map = affine_map<(d0, d1, d2, d3, d4) -> (d1 + d2, d3 + d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
#map3 = affine_map<(d0) -> ()>
#map4 = affine_map<(d0) -> (d0)>
#map5 = affine_map<(d0, d1, d2) -> (d0)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map7 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2 + d3, d4 + d5)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d5)>
#map9 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4)>
#map10 = affine_map<(d0, d1, d2) -> ()>
#map11 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2 * 2 + d3, d4 * 2 + d5)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d0, d2 * 2, d3 * 2)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d1, d0)>
#map14 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map15 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map16 = affine_map<(d0, d1) -> (d1, d0)>
#map17 = affine_map<(d0, d1) -> (d0, d1)>
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<16x1x3x3xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x16x3x3xf32>, %arg4: tensor<16xf32>, %arg5: tensor<16xf32>, %arg6: tensor<16x16x3x3xf32>, %arg7: tensor<16xf32>, %arg8: tensor<16xf32>, %arg9: tensor<16x16x3x3xf32>, %arg10: tensor<16xf32>, %arg11: tensor<16xf32>, %arg12: tensor<16x16x3x3xf32>, %arg13: tensor<16xf32>, %arg14: tensor<16xf32>, %arg15: tensor<32x16x3x3xf32>, %arg16: tensor<32xf32>, %arg17: tensor<32xf32>, %arg18: tensor<32x32x3x3xf32>, %arg19: tensor<32xf32>, %arg20: tensor<32xf32>, %arg21: tensor<32x16x1x1xf32>, %arg22: tensor<32xf32>, %arg23: tensor<32xf32>, %arg24: tensor<32x32x3x3xf32>, %arg25: tensor<32xf32>, %arg26: tensor<32xf32>, %arg27: tensor<32x32x3x3xf32>, %arg28: tensor<32xf32>, %arg29: tensor<32xf32>, %arg30: tensor<64x32x3x3xf32>, %arg31: tensor<64xf32>, %arg32: tensor<64xf32>, %arg33: tensor<64x64x3x3xf32>, %arg34: tensor<64xf32>, %arg35: tensor<64xf32>, %arg36: tensor<64x32x1x1xf32>, %arg37: tensor<64xf32>, %arg38: tensor<64xf32>, %arg39: tensor<64x64x3x3xf32>, %arg40: tensor<64xf32>, %arg41: tensor<64xf32>, %arg42: tensor<64x64x3x3xf32>, %arg43: tensor<64xf32>, %arg44: tensor<64xf32>, %arg45: tensor<10x64xf32>, %arg46: tensor<10xf32>, %arg47: tensor<16xf32>, %arg48: tensor<16xf32>, %arg49: tensor<16xf32>, %arg50: tensor<16xf32>, %arg51: tensor<16xf32>, %arg52: tensor<16xf32>, %arg53: tensor<16xf32>, %arg54: tensor<16xf32>, %arg55: tensor<16xf32>, %arg56: tensor<16xf32>, %arg57: tensor<32xf32>, %arg58: tensor<32xf32>, %arg59: tensor<32xf32>, %arg60: tensor<32xf32>, %arg61: tensor<32xf32>, %arg62: tensor<32xf32>, %arg63: tensor<32xf32>, %arg64: tensor<32xf32>, %arg65: tensor<32xf32>, %arg66: tensor<32xf32>, %arg67: tensor<64xf32>, %arg68: tensor<64xf32>, %arg69: tensor<64xf32>, %arg70: tensor<64xf32>, %arg71: tensor<64xf32>, %arg72: tensor<64xf32>, %arg73: tensor<64xf32>, %arg74: tensor<64xf32>, %arg75: tensor<64xf32>, %arg76: tensor<64xf32>, %arg77: tensor<1x1x28x28xf32>{secret.secret}) -> (tensor<1x10xf32> {jax.result_info = "result[0]"}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x7x7xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x14x14xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<16x28x28xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x10xf32>
    %cst_4 = arith.constant dense<4.900000e+01> : tensor<f32>
    %cst_5 = arith.constant dense<1.000000e+00> : tensor<f32>
    %cst_6 = arith.constant dense<9.99999974E-6> : tensor<f32>
    %cst_7 = arith.constant 0.000000e+00 : f32
    %collapsed = tensor.collapse_shape %arg77 [[0, 1, 2], [3]] : tensor<1x1x28x28xf32> into tensor<28x28xf32>
    %padded = tensor.pad %collapsed low[1, 1] high[1, 1] {
    ^bb0(%arg78: index, %arg79: index):
      tensor.yield %cst_7 : f32
    } : tensor<28x28xf32> to tensor<30x30xf32>
    %collapsed_8 = tensor.collapse_shape %arg0 [[0, 1], [2], [3]] : tensor<16x1x3x3xf32> into tensor<16x3x3xf32>
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "reduction"]} ins(%padded, %collapsed_8 : tensor<30x30xf32>, tensor<16x3x3xf32>) outs(%cst_2 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %in_168: f32, %out: f32):
      %310 = arith.mulf %in, %in_168 : f32
      %311 = arith.addf %out, %310 : f32
      linalg.yield %311 : f32
    } -> tensor<16x28x28xf32>
    %expanded = tensor.expand_shape %0 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %expanded_9 = tensor.expand_shape %arg48 [[0, 1, 2, 3]] output_shape [1, 16, 1, 1] : tensor<16xf32> into tensor<1x16x1x1xf32>
    %1 = tensor.empty() : tensor<16xf32>
    %2 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel"]} ins(%cst_6 : tensor<f32>) outs(%1 : tensor<16xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16xf32>
    %expanded_10 = tensor.expand_shape %2 [[0, 1, 2, 3]] output_shape [1, 16, 1, 1] : tensor<16xf32> into tensor<1x16x1x1xf32>
    %3 = arith.addf %expanded_9, %expanded_10 : tensor<1x16x1x1xf32>
    %4 = math.rsqrt %3 : tensor<1x16x1x1xf32>
    %5 = tensor.empty() : tensor<16x28x28xf32>
    %6 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg47 : tensor<16xf32>) outs(%5 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x28x28xf32>
    %expanded_11 = tensor.expand_shape %6 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %7 = arith.subf %expanded, %expanded_11 : tensor<1x16x28x28xf32>
    %collapsed_12 = tensor.collapse_shape %4 [[0, 1, 2, 3]] : tensor<1x16x1x1xf32> into tensor<16xf32>
    %8 = tensor.empty() : tensor<16x28x28xf32>
    %9 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_12 : tensor<16xf32>) outs(%8 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x28x28xf32>
    %expanded_13 = tensor.expand_shape %9 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %10 = arith.mulf %7, %expanded_13 : tensor<1x16x28x28xf32>
    %11 = tensor.empty() : tensor<16x28x28xf32>
    %12 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1 : tensor<16xf32>) outs(%11 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x28x28xf32>
    %expanded_14 = tensor.expand_shape %12 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %13 = arith.mulf %10, %expanded_14 : tensor<1x16x28x28xf32>
    %14 = tensor.empty() : tensor<16x28x28xf32>
    %15 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg2 : tensor<16xf32>) outs(%14 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x28x28xf32>
    %expanded_15 = tensor.expand_shape %15 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %16 = arith.addf %13, %expanded_15 : tensor<1x16x28x28xf32>
    %17 = call @relu(%16) : (tensor<1x16x28x28xf32>) -> tensor<1x16x28x28xf32>
    %collapsed_16 = tensor.collapse_shape %17 [[0, 1], [2], [3]] : tensor<1x16x28x28xf32> into tensor<16x28x28xf32>
    %padded_17 = tensor.pad %collapsed_16 low[0, 1, 1] high[0, 1, 1] {
    ^bb0(%arg78: index, %arg79: index, %arg80: index):
      tensor.yield %cst_7 : f32
    } : tensor<16x28x28xf32> to tensor<16x30x30xf32>
    %18 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel", "reduction"]} ins(%padded_17, %arg3 : tensor<16x30x30xf32>, tensor<16x16x3x3xf32>) outs(%cst_2 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %in_168: f32, %out: f32):
      %310 = arith.mulf %in, %in_168 : f32
      %311 = arith.addf %out, %310 : f32
      linalg.yield %311 : f32
    } -> tensor<16x28x28xf32>
    %expanded_18 = tensor.expand_shape %18 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %expanded_19 = tensor.expand_shape %arg50 [[0, 1, 2, 3]] output_shape [1, 16, 1, 1] : tensor<16xf32> into tensor<1x16x1x1xf32>
    %19 = tensor.empty() : tensor<16xf32>
    %20 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel"]} ins(%cst_6 : tensor<f32>) outs(%19 : tensor<16xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16xf32>
    %expanded_20 = tensor.expand_shape %20 [[0, 1, 2, 3]] output_shape [1, 16, 1, 1] : tensor<16xf32> into tensor<1x16x1x1xf32>
    %21 = arith.addf %expanded_19, %expanded_20 : tensor<1x16x1x1xf32>
    %22 = math.rsqrt %21 : tensor<1x16x1x1xf32>
    %23 = tensor.empty() : tensor<16x28x28xf32>
    %24 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg49 : tensor<16xf32>) outs(%23 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x28x28xf32>
    %expanded_21 = tensor.expand_shape %24 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %25 = arith.subf %expanded_18, %expanded_21 : tensor<1x16x28x28xf32>
    %collapsed_22 = tensor.collapse_shape %22 [[0, 1, 2, 3]] : tensor<1x16x1x1xf32> into tensor<16xf32>
    %26 = tensor.empty() : tensor<16x28x28xf32>
    %27 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_22 : tensor<16xf32>) outs(%26 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x28x28xf32>
    %expanded_23 = tensor.expand_shape %27 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %28 = arith.mulf %25, %expanded_23 : tensor<1x16x28x28xf32>
    %29 = tensor.empty() : tensor<16x28x28xf32>
    %30 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg4 : tensor<16xf32>) outs(%29 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x28x28xf32>
    %expanded_24 = tensor.expand_shape %30 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %31 = arith.mulf %28, %expanded_24 : tensor<1x16x28x28xf32>
    %32 = tensor.empty() : tensor<16x28x28xf32>
    %33 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg5 : tensor<16xf32>) outs(%32 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x28x28xf32>
    %expanded_25 = tensor.expand_shape %33 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %34 = arith.addf %31, %expanded_25 : tensor<1x16x28x28xf32>
    %35 = call @relu(%34) : (tensor<1x16x28x28xf32>) -> tensor<1x16x28x28xf32>
    %collapsed_26 = tensor.collapse_shape %35 [[0, 1], [2], [3]] : tensor<1x16x28x28xf32> into tensor<16x28x28xf32>
    %padded_27 = tensor.pad %collapsed_26 low[0, 1, 1] high[0, 1, 1] {
    ^bb0(%arg78: index, %arg79: index, %arg80: index):
      tensor.yield %cst_7 : f32
    } : tensor<16x28x28xf32> to tensor<16x30x30xf32>
    %36 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel", "reduction"]} ins(%padded_27, %arg6 : tensor<16x30x30xf32>, tensor<16x16x3x3xf32>) outs(%cst_2 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %in_168: f32, %out: f32):
      %310 = arith.mulf %in, %in_168 : f32
      %311 = arith.addf %out, %310 : f32
      linalg.yield %311 : f32
    } -> tensor<16x28x28xf32>
    %expanded_28 = tensor.expand_shape %36 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %expanded_29 = tensor.expand_shape %arg52 [[0, 1, 2, 3]] output_shape [1, 16, 1, 1] : tensor<16xf32> into tensor<1x16x1x1xf32>
    %37 = tensor.empty() : tensor<16xf32>
    %38 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel"]} ins(%cst_6 : tensor<f32>) outs(%37 : tensor<16xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16xf32>
    %expanded_30 = tensor.expand_shape %38 [[0, 1, 2, 3]] output_shape [1, 16, 1, 1] : tensor<16xf32> into tensor<1x16x1x1xf32>
    %39 = arith.addf %expanded_29, %expanded_30 : tensor<1x16x1x1xf32>
    %40 = math.rsqrt %39 : tensor<1x16x1x1xf32>
    %41 = tensor.empty() : tensor<16x28x28xf32>
    %42 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg51 : tensor<16xf32>) outs(%41 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x28x28xf32>
    %expanded_31 = tensor.expand_shape %42 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %43 = arith.subf %expanded_28, %expanded_31 : tensor<1x16x28x28xf32>
    %collapsed_32 = tensor.collapse_shape %40 [[0, 1, 2, 3]] : tensor<1x16x1x1xf32> into tensor<16xf32>
    %44 = tensor.empty() : tensor<16x28x28xf32>
    %45 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_32 : tensor<16xf32>) outs(%44 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x28x28xf32>
    %expanded_33 = tensor.expand_shape %45 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %46 = arith.mulf %43, %expanded_33 : tensor<1x16x28x28xf32>
    %47 = tensor.empty() : tensor<16x28x28xf32>
    %48 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7 : tensor<16xf32>) outs(%47 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x28x28xf32>
    %expanded_34 = tensor.expand_shape %48 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %49 = arith.mulf %46, %expanded_34 : tensor<1x16x28x28xf32>
    %50 = tensor.empty() : tensor<16x28x28xf32>
    %51 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8 : tensor<16xf32>) outs(%50 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x28x28xf32>
    %expanded_35 = tensor.expand_shape %51 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %52 = arith.addf %49, %expanded_35 : tensor<1x16x28x28xf32>
    %53 = tensor.empty() : tensor<16x28x28xf32>
    %54 = linalg.generic {indexing_maps = [#map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_5 : tensor<f32>) outs(%53 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x28x28xf32>
    %expanded_36 = tensor.expand_shape %54 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %55 = arith.mulf %17, %expanded_36 : tensor<1x16x28x28xf32>
    %56 = arith.addf %52, %55 : tensor<1x16x28x28xf32>
    %57 = call @relu(%56) : (tensor<1x16x28x28xf32>) -> tensor<1x16x28x28xf32>
    %collapsed_37 = tensor.collapse_shape %57 [[0, 1], [2], [3]] : tensor<1x16x28x28xf32> into tensor<16x28x28xf32>
    %padded_38 = tensor.pad %collapsed_37 low[0, 1, 1] high[0, 1, 1] {
    ^bb0(%arg78: index, %arg79: index, %arg80: index):
      tensor.yield %cst_7 : f32
    } : tensor<16x28x28xf32> to tensor<16x30x30xf32>
    %58 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel", "reduction"]} ins(%padded_38, %arg9 : tensor<16x30x30xf32>, tensor<16x16x3x3xf32>) outs(%cst_2 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %in_168: f32, %out: f32):
      %310 = arith.mulf %in, %in_168 : f32
      %311 = arith.addf %out, %310 : f32
      linalg.yield %311 : f32
    } -> tensor<16x28x28xf32>
    %expanded_39 = tensor.expand_shape %58 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %expanded_40 = tensor.expand_shape %arg54 [[0, 1, 2, 3]] output_shape [1, 16, 1, 1] : tensor<16xf32> into tensor<1x16x1x1xf32>
    %59 = tensor.empty() : tensor<16xf32>
    %60 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel"]} ins(%cst_6 : tensor<f32>) outs(%59 : tensor<16xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16xf32>
    %expanded_41 = tensor.expand_shape %60 [[0, 1, 2, 3]] output_shape [1, 16, 1, 1] : tensor<16xf32> into tensor<1x16x1x1xf32>
    %61 = arith.addf %expanded_40, %expanded_41 : tensor<1x16x1x1xf32>
    %62 = math.rsqrt %61 : tensor<1x16x1x1xf32>
    %63 = tensor.empty() : tensor<16x28x28xf32>
    %64 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg53 : tensor<16xf32>) outs(%63 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x28x28xf32>
    %expanded_42 = tensor.expand_shape %64 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %65 = arith.subf %expanded_39, %expanded_42 : tensor<1x16x28x28xf32>
    %collapsed_43 = tensor.collapse_shape %62 [[0, 1, 2, 3]] : tensor<1x16x1x1xf32> into tensor<16xf32>
    %66 = tensor.empty() : tensor<16x28x28xf32>
    %67 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_43 : tensor<16xf32>) outs(%66 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x28x28xf32>
    %expanded_44 = tensor.expand_shape %67 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %68 = arith.mulf %65, %expanded_44 : tensor<1x16x28x28xf32>
    %69 = tensor.empty() : tensor<16x28x28xf32>
    %70 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10 : tensor<16xf32>) outs(%69 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x28x28xf32>
    %expanded_45 = tensor.expand_shape %70 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %71 = arith.mulf %68, %expanded_45 : tensor<1x16x28x28xf32>
    %72 = tensor.empty() : tensor<16x28x28xf32>
    %73 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11 : tensor<16xf32>) outs(%72 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x28x28xf32>
    %expanded_46 = tensor.expand_shape %73 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %74 = arith.addf %71, %expanded_46 : tensor<1x16x28x28xf32>
    %75 = call @relu(%74) : (tensor<1x16x28x28xf32>) -> tensor<1x16x28x28xf32>
    %collapsed_47 = tensor.collapse_shape %75 [[0, 1], [2], [3]] : tensor<1x16x28x28xf32> into tensor<16x28x28xf32>
    %padded_48 = tensor.pad %collapsed_47 low[0, 1, 1] high[0, 1, 1] {
    ^bb0(%arg78: index, %arg79: index, %arg80: index):
      tensor.yield %cst_7 : f32
    } : tensor<16x28x28xf32> to tensor<16x30x30xf32>
    %76 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel", "reduction"]} ins(%padded_48, %arg12 : tensor<16x30x30xf32>, tensor<16x16x3x3xf32>) outs(%cst_2 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %in_168: f32, %out: f32):
      %310 = arith.mulf %in, %in_168 : f32
      %311 = arith.addf %out, %310 : f32
      linalg.yield %311 : f32
    } -> tensor<16x28x28xf32>
    %expanded_49 = tensor.expand_shape %76 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %expanded_50 = tensor.expand_shape %arg56 [[0, 1, 2, 3]] output_shape [1, 16, 1, 1] : tensor<16xf32> into tensor<1x16x1x1xf32>
    %77 = tensor.empty() : tensor<16xf32>
    %78 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel"]} ins(%cst_6 : tensor<f32>) outs(%77 : tensor<16xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16xf32>
    %expanded_51 = tensor.expand_shape %78 [[0, 1, 2, 3]] output_shape [1, 16, 1, 1] : tensor<16xf32> into tensor<1x16x1x1xf32>
    %79 = arith.addf %expanded_50, %expanded_51 : tensor<1x16x1x1xf32>
    %80 = math.rsqrt %79 : tensor<1x16x1x1xf32>
    %81 = tensor.empty() : tensor<16x28x28xf32>
    %82 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg55 : tensor<16xf32>) outs(%81 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x28x28xf32>
    %expanded_52 = tensor.expand_shape %82 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %83 = arith.subf %expanded_49, %expanded_52 : tensor<1x16x28x28xf32>
    %collapsed_53 = tensor.collapse_shape %80 [[0, 1, 2, 3]] : tensor<1x16x1x1xf32> into tensor<16xf32>
    %84 = tensor.empty() : tensor<16x28x28xf32>
    %85 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_53 : tensor<16xf32>) outs(%84 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x28x28xf32>
    %expanded_54 = tensor.expand_shape %85 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %86 = arith.mulf %83, %expanded_54 : tensor<1x16x28x28xf32>
    %87 = tensor.empty() : tensor<16x28x28xf32>
    %88 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg13 : tensor<16xf32>) outs(%87 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x28x28xf32>
    %expanded_55 = tensor.expand_shape %88 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %89 = arith.mulf %86, %expanded_55 : tensor<1x16x28x28xf32>
    %90 = tensor.empty() : tensor<16x28x28xf32>
    %91 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg14 : tensor<16xf32>) outs(%90 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x28x28xf32>
    %expanded_56 = tensor.expand_shape %91 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %92 = arith.addf %89, %expanded_56 : tensor<1x16x28x28xf32>
    %93 = tensor.empty() : tensor<16x28x28xf32>
    %94 = linalg.generic {indexing_maps = [#map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_5 : tensor<f32>) outs(%93 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x28x28xf32>
    %expanded_57 = tensor.expand_shape %94 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %95 = arith.mulf %57, %expanded_57 : tensor<1x16x28x28xf32>
    %96 = arith.addf %92, %95 : tensor<1x16x28x28xf32>
    %97 = call @relu(%96) : (tensor<1x16x28x28xf32>) -> tensor<1x16x28x28xf32>
    %collapsed_58 = tensor.collapse_shape %97 [[0, 1], [2], [3]] : tensor<1x16x28x28xf32> into tensor<16x28x28xf32>
    %padded_59 = tensor.pad %collapsed_58 low[0, 1, 1] high[0, 1, 1] {
    ^bb0(%arg78: index, %arg79: index, %arg80: index):
      tensor.yield %cst_7 : f32
    } : tensor<16x28x28xf32> to tensor<16x30x30xf32>
    %98 = linalg.generic {indexing_maps = [#map11, #map8, #map9], iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel", "reduction"]} ins(%padded_59, %arg15 : tensor<16x30x30xf32>, tensor<32x16x3x3xf32>) outs(%cst_1 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %in_168: f32, %out: f32):
      %310 = arith.mulf %in, %in_168 : f32
      %311 = arith.addf %out, %310 : f32
      linalg.yield %311 : f32
    } -> tensor<32x14x14xf32>
    %expanded_60 = tensor.expand_shape %98 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %expanded_61 = tensor.expand_shape %arg58 [[0, 1, 2, 3]] output_shape [1, 32, 1, 1] : tensor<32xf32> into tensor<1x32x1x1xf32>
    %99 = tensor.empty() : tensor<32xf32>
    %100 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel"]} ins(%cst_6 : tensor<f32>) outs(%99 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32xf32>
    %expanded_62 = tensor.expand_shape %100 [[0, 1, 2, 3]] output_shape [1, 32, 1, 1] : tensor<32xf32> into tensor<1x32x1x1xf32>
    %101 = arith.addf %expanded_61, %expanded_62 : tensor<1x32x1x1xf32>
    %102 = math.rsqrt %101 : tensor<1x32x1x1xf32>
    %103 = tensor.empty() : tensor<32x14x14xf32>
    %104 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg57 : tensor<32xf32>) outs(%103 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x14x14xf32>
    %expanded_63 = tensor.expand_shape %104 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %105 = arith.subf %expanded_60, %expanded_63 : tensor<1x32x14x14xf32>
    %collapsed_64 = tensor.collapse_shape %102 [[0, 1, 2, 3]] : tensor<1x32x1x1xf32> into tensor<32xf32>
    %106 = tensor.empty() : tensor<32x14x14xf32>
    %107 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_64 : tensor<32xf32>) outs(%106 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x14x14xf32>
    %expanded_65 = tensor.expand_shape %107 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %108 = arith.mulf %105, %expanded_65 : tensor<1x32x14x14xf32>
    %109 = tensor.empty() : tensor<32x14x14xf32>
    %110 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg16 : tensor<32xf32>) outs(%109 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x14x14xf32>
    %expanded_66 = tensor.expand_shape %110 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %111 = arith.mulf %108, %expanded_66 : tensor<1x32x14x14xf32>
    %112 = tensor.empty() : tensor<32x14x14xf32>
    %113 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg17 : tensor<32xf32>) outs(%112 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x14x14xf32>
    %expanded_67 = tensor.expand_shape %113 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %114 = arith.addf %111, %expanded_67 : tensor<1x32x14x14xf32>
    %115 = call @relu_16(%114) : (tensor<1x32x14x14xf32>) -> tensor<1x32x14x14xf32>
    %collapsed_68 = tensor.collapse_shape %115 [[0, 1], [2], [3]] : tensor<1x32x14x14xf32> into tensor<32x14x14xf32>
    %padded_69 = tensor.pad %collapsed_68 low[0, 1, 1] high[0, 1, 1] {
    ^bb0(%arg78: index, %arg79: index, %arg80: index):
      tensor.yield %cst_7 : f32
    } : tensor<32x14x14xf32> to tensor<32x16x16xf32>
    %116 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel", "reduction"]} ins(%padded_69, %arg18 : tensor<32x16x16xf32>, tensor<32x32x3x3xf32>) outs(%cst_1 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %in_168: f32, %out: f32):
      %310 = arith.mulf %in, %in_168 : f32
      %311 = arith.addf %out, %310 : f32
      linalg.yield %311 : f32
    } -> tensor<32x14x14xf32>
    %expanded_70 = tensor.expand_shape %116 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %expanded_71 = tensor.expand_shape %arg60 [[0, 1, 2, 3]] output_shape [1, 32, 1, 1] : tensor<32xf32> into tensor<1x32x1x1xf32>
    %117 = tensor.empty() : tensor<32xf32>
    %118 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel"]} ins(%cst_6 : tensor<f32>) outs(%117 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32xf32>
    %expanded_72 = tensor.expand_shape %118 [[0, 1, 2, 3]] output_shape [1, 32, 1, 1] : tensor<32xf32> into tensor<1x32x1x1xf32>
    %119 = arith.addf %expanded_71, %expanded_72 : tensor<1x32x1x1xf32>
    %120 = math.rsqrt %119 : tensor<1x32x1x1xf32>
    %121 = tensor.empty() : tensor<32x14x14xf32>
    %122 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg59 : tensor<32xf32>) outs(%121 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x14x14xf32>
    %expanded_73 = tensor.expand_shape %122 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %123 = arith.subf %expanded_70, %expanded_73 : tensor<1x32x14x14xf32>
    %collapsed_74 = tensor.collapse_shape %120 [[0, 1, 2, 3]] : tensor<1x32x1x1xf32> into tensor<32xf32>
    %124 = tensor.empty() : tensor<32x14x14xf32>
    %125 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_74 : tensor<32xf32>) outs(%124 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x14x14xf32>
    %expanded_75 = tensor.expand_shape %125 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %126 = arith.mulf %123, %expanded_75 : tensor<1x32x14x14xf32>
    %127 = tensor.empty() : tensor<32x14x14xf32>
    %128 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg19 : tensor<32xf32>) outs(%127 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x14x14xf32>
    %expanded_76 = tensor.expand_shape %128 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %129 = arith.mulf %126, %expanded_76 : tensor<1x32x14x14xf32>
    %130 = tensor.empty() : tensor<32x14x14xf32>
    %131 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg20 : tensor<32xf32>) outs(%130 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x14x14xf32>
    %expanded_77 = tensor.expand_shape %131 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %132 = arith.addf %129, %expanded_77 : tensor<1x32x14x14xf32>
    %collapsed_78 = tensor.collapse_shape %97 [[0, 1], [2], [3]] : tensor<1x16x28x28xf32> into tensor<16x28x28xf32>
    %collapsed_79 = tensor.collapse_shape %arg21 [[0], [1, 2, 3]] : tensor<32x16x1x1xf32> into tensor<32x16xf32>
    %133 = linalg.generic {indexing_maps = [#map12, #map13, #map14], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%collapsed_78, %collapsed_79 : tensor<16x28x28xf32>, tensor<32x16xf32>) outs(%cst_1 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %in_168: f32, %out: f32):
      %310 = arith.mulf %in, %in_168 : f32
      %311 = arith.addf %out, %310 : f32
      linalg.yield %311 : f32
    } -> tensor<32x14x14xf32>
    %expanded_80 = tensor.expand_shape %133 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %expanded_81 = tensor.expand_shape %arg62 [[0, 1, 2, 3]] output_shape [1, 32, 1, 1] : tensor<32xf32> into tensor<1x32x1x1xf32>
    %134 = tensor.empty() : tensor<32xf32>
    %135 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel"]} ins(%cst_6 : tensor<f32>) outs(%134 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32xf32>
    %expanded_82 = tensor.expand_shape %135 [[0, 1, 2, 3]] output_shape [1, 32, 1, 1] : tensor<32xf32> into tensor<1x32x1x1xf32>
    %136 = arith.addf %expanded_81, %expanded_82 : tensor<1x32x1x1xf32>
    %137 = math.rsqrt %136 : tensor<1x32x1x1xf32>
    %138 = tensor.empty() : tensor<32x14x14xf32>
    %139 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg61 : tensor<32xf32>) outs(%138 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x14x14xf32>
    %expanded_83 = tensor.expand_shape %139 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %140 = arith.subf %expanded_80, %expanded_83 : tensor<1x32x14x14xf32>
    %collapsed_84 = tensor.collapse_shape %137 [[0, 1, 2, 3]] : tensor<1x32x1x1xf32> into tensor<32xf32>
    %141 = tensor.empty() : tensor<32x14x14xf32>
    %142 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_84 : tensor<32xf32>) outs(%141 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x14x14xf32>
    %expanded_85 = tensor.expand_shape %142 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %143 = arith.mulf %140, %expanded_85 : tensor<1x32x14x14xf32>
    %144 = tensor.empty() : tensor<32x14x14xf32>
    %145 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg22 : tensor<32xf32>) outs(%144 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x14x14xf32>
    %expanded_86 = tensor.expand_shape %145 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %146 = arith.mulf %143, %expanded_86 : tensor<1x32x14x14xf32>
    %147 = tensor.empty() : tensor<32x14x14xf32>
    %148 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg23 : tensor<32xf32>) outs(%147 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x14x14xf32>
    %expanded_87 = tensor.expand_shape %148 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %149 = arith.addf %146, %expanded_87 : tensor<1x32x14x14xf32>
    %150 = tensor.empty() : tensor<32x14x14xf32>
    %151 = linalg.generic {indexing_maps = [#map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_5 : tensor<f32>) outs(%150 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x14x14xf32>
    %expanded_88 = tensor.expand_shape %151 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %152 = arith.mulf %149, %expanded_88 : tensor<1x32x14x14xf32>
    %153 = arith.addf %132, %152 : tensor<1x32x14x14xf32>
    %154 = call @relu_16(%153) : (tensor<1x32x14x14xf32>) -> tensor<1x32x14x14xf32>
    %collapsed_89 = tensor.collapse_shape %154 [[0, 1], [2], [3]] : tensor<1x32x14x14xf32> into tensor<32x14x14xf32>
    %padded_90 = tensor.pad %collapsed_89 low[0, 1, 1] high[0, 1, 1] {
    ^bb0(%arg78: index, %arg79: index, %arg80: index):
      tensor.yield %cst_7 : f32
    } : tensor<32x14x14xf32> to tensor<32x16x16xf32>
    %155 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel", "reduction"]} ins(%padded_90, %arg24 : tensor<32x16x16xf32>, tensor<32x32x3x3xf32>) outs(%cst_1 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %in_168: f32, %out: f32):
      %310 = arith.mulf %in, %in_168 : f32
      %311 = arith.addf %out, %310 : f32
      linalg.yield %311 : f32
    } -> tensor<32x14x14xf32>
    %expanded_91 = tensor.expand_shape %155 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %expanded_92 = tensor.expand_shape %arg64 [[0, 1, 2, 3]] output_shape [1, 32, 1, 1] : tensor<32xf32> into tensor<1x32x1x1xf32>
    %156 = tensor.empty() : tensor<32xf32>
    %157 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel"]} ins(%cst_6 : tensor<f32>) outs(%156 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32xf32>
    %expanded_93 = tensor.expand_shape %157 [[0, 1, 2, 3]] output_shape [1, 32, 1, 1] : tensor<32xf32> into tensor<1x32x1x1xf32>
    %158 = arith.addf %expanded_92, %expanded_93 : tensor<1x32x1x1xf32>
    %159 = math.rsqrt %158 : tensor<1x32x1x1xf32>
    %160 = tensor.empty() : tensor<32x14x14xf32>
    %161 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg63 : tensor<32xf32>) outs(%160 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x14x14xf32>
    %expanded_94 = tensor.expand_shape %161 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %162 = arith.subf %expanded_91, %expanded_94 : tensor<1x32x14x14xf32>
    %collapsed_95 = tensor.collapse_shape %159 [[0, 1, 2, 3]] : tensor<1x32x1x1xf32> into tensor<32xf32>
    %163 = tensor.empty() : tensor<32x14x14xf32>
    %164 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_95 : tensor<32xf32>) outs(%163 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x14x14xf32>
    %expanded_96 = tensor.expand_shape %164 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %165 = arith.mulf %162, %expanded_96 : tensor<1x32x14x14xf32>
    %166 = tensor.empty() : tensor<32x14x14xf32>
    %167 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg25 : tensor<32xf32>) outs(%166 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x14x14xf32>
    %expanded_97 = tensor.expand_shape %167 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %168 = arith.mulf %165, %expanded_97 : tensor<1x32x14x14xf32>
    %169 = tensor.empty() : tensor<32x14x14xf32>
    %170 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg26 : tensor<32xf32>) outs(%169 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x14x14xf32>
    %expanded_98 = tensor.expand_shape %170 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %171 = arith.addf %168, %expanded_98 : tensor<1x32x14x14xf32>
    %172 = call @relu_16(%171) : (tensor<1x32x14x14xf32>) -> tensor<1x32x14x14xf32>
    %collapsed_99 = tensor.collapse_shape %172 [[0, 1], [2], [3]] : tensor<1x32x14x14xf32> into tensor<32x14x14xf32>
    %padded_100 = tensor.pad %collapsed_99 low[0, 1, 1] high[0, 1, 1] {
    ^bb0(%arg78: index, %arg79: index, %arg80: index):
      tensor.yield %cst_7 : f32
    } : tensor<32x14x14xf32> to tensor<32x16x16xf32>
    %173 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel", "reduction"]} ins(%padded_100, %arg27 : tensor<32x16x16xf32>, tensor<32x32x3x3xf32>) outs(%cst_1 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %in_168: f32, %out: f32):
      %310 = arith.mulf %in, %in_168 : f32
      %311 = arith.addf %out, %310 : f32
      linalg.yield %311 : f32
    } -> tensor<32x14x14xf32>
    %expanded_101 = tensor.expand_shape %173 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %expanded_102 = tensor.expand_shape %arg66 [[0, 1, 2, 3]] output_shape [1, 32, 1, 1] : tensor<32xf32> into tensor<1x32x1x1xf32>
    %174 = tensor.empty() : tensor<32xf32>
    %175 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel"]} ins(%cst_6 : tensor<f32>) outs(%174 : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32xf32>
    %expanded_103 = tensor.expand_shape %175 [[0, 1, 2, 3]] output_shape [1, 32, 1, 1] : tensor<32xf32> into tensor<1x32x1x1xf32>
    %176 = arith.addf %expanded_102, %expanded_103 : tensor<1x32x1x1xf32>
    %177 = math.rsqrt %176 : tensor<1x32x1x1xf32>
    %178 = tensor.empty() : tensor<32x14x14xf32>
    %179 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg65 : tensor<32xf32>) outs(%178 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x14x14xf32>
    %expanded_104 = tensor.expand_shape %179 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %180 = arith.subf %expanded_101, %expanded_104 : tensor<1x32x14x14xf32>
    %collapsed_105 = tensor.collapse_shape %177 [[0, 1, 2, 3]] : tensor<1x32x1x1xf32> into tensor<32xf32>
    %181 = tensor.empty() : tensor<32x14x14xf32>
    %182 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_105 : tensor<32xf32>) outs(%181 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x14x14xf32>
    %expanded_106 = tensor.expand_shape %182 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %183 = arith.mulf %180, %expanded_106 : tensor<1x32x14x14xf32>
    %184 = tensor.empty() : tensor<32x14x14xf32>
    %185 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg28 : tensor<32xf32>) outs(%184 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x14x14xf32>
    %expanded_107 = tensor.expand_shape %185 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %186 = arith.mulf %183, %expanded_107 : tensor<1x32x14x14xf32>
    %187 = tensor.empty() : tensor<32x14x14xf32>
    %188 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg29 : tensor<32xf32>) outs(%187 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x14x14xf32>
    %expanded_108 = tensor.expand_shape %188 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %189 = arith.addf %186, %expanded_108 : tensor<1x32x14x14xf32>
    %190 = tensor.empty() : tensor<32x14x14xf32>
    %191 = linalg.generic {indexing_maps = [#map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_5 : tensor<f32>) outs(%190 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x14x14xf32>
    %expanded_109 = tensor.expand_shape %191 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %192 = arith.mulf %154, %expanded_109 : tensor<1x32x14x14xf32>
    %193 = arith.addf %189, %192 : tensor<1x32x14x14xf32>
    %194 = call @relu_16(%193) : (tensor<1x32x14x14xf32>) -> tensor<1x32x14x14xf32>
    %collapsed_110 = tensor.collapse_shape %194 [[0, 1], [2], [3]] : tensor<1x32x14x14xf32> into tensor<32x14x14xf32>
    %padded_111 = tensor.pad %collapsed_110 low[0, 1, 1] high[0, 1, 1] {
    ^bb0(%arg78: index, %arg79: index, %arg80: index):
      tensor.yield %cst_7 : f32
    } : tensor<32x14x14xf32> to tensor<32x16x16xf32>
    %195 = linalg.generic {indexing_maps = [#map11, #map8, #map9], iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel", "reduction"]} ins(%padded_111, %arg30 : tensor<32x16x16xf32>, tensor<64x32x3x3xf32>) outs(%cst_0 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %in_168: f32, %out: f32):
      %310 = arith.mulf %in, %in_168 : f32
      %311 = arith.addf %out, %310 : f32
      linalg.yield %311 : f32
    } -> tensor<64x7x7xf32>
    %expanded_112 = tensor.expand_shape %195 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %expanded_113 = tensor.expand_shape %arg68 [[0, 1, 2, 3]] output_shape [1, 64, 1, 1] : tensor<64xf32> into tensor<1x64x1x1xf32>
    %196 = tensor.empty() : tensor<64xf32>
    %197 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel"]} ins(%cst_6 : tensor<f32>) outs(%196 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64xf32>
    %expanded_114 = tensor.expand_shape %197 [[0, 1, 2, 3]] output_shape [1, 64, 1, 1] : tensor<64xf32> into tensor<1x64x1x1xf32>
    %198 = arith.addf %expanded_113, %expanded_114 : tensor<1x64x1x1xf32>
    %199 = math.rsqrt %198 : tensor<1x64x1x1xf32>
    %200 = tensor.empty() : tensor<64x7x7xf32>
    %201 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg67 : tensor<64xf32>) outs(%200 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x7x7xf32>
    %expanded_115 = tensor.expand_shape %201 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %202 = arith.subf %expanded_112, %expanded_115 : tensor<1x64x7x7xf32>
    %collapsed_116 = tensor.collapse_shape %199 [[0, 1, 2, 3]] : tensor<1x64x1x1xf32> into tensor<64xf32>
    %203 = tensor.empty() : tensor<64x7x7xf32>
    %204 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_116 : tensor<64xf32>) outs(%203 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x7x7xf32>
    %expanded_117 = tensor.expand_shape %204 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %205 = arith.mulf %202, %expanded_117 : tensor<1x64x7x7xf32>
    %206 = tensor.empty() : tensor<64x7x7xf32>
    %207 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg31 : tensor<64xf32>) outs(%206 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x7x7xf32>
    %expanded_118 = tensor.expand_shape %207 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %208 = arith.mulf %205, %expanded_118 : tensor<1x64x7x7xf32>
    %209 = tensor.empty() : tensor<64x7x7xf32>
    %210 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg32 : tensor<64xf32>) outs(%209 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x7x7xf32>
    %expanded_119 = tensor.expand_shape %210 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %211 = arith.addf %208, %expanded_119 : tensor<1x64x7x7xf32>
    %212 = call @relu_33(%211) : (tensor<1x64x7x7xf32>) -> tensor<1x64x7x7xf32>
    %collapsed_120 = tensor.collapse_shape %212 [[0, 1], [2], [3]] : tensor<1x64x7x7xf32> into tensor<64x7x7xf32>
    %padded_121 = tensor.pad %collapsed_120 low[0, 1, 1] high[0, 1, 1] {
    ^bb0(%arg78: index, %arg79: index, %arg80: index):
      tensor.yield %cst_7 : f32
    } : tensor<64x7x7xf32> to tensor<64x9x9xf32>
    %213 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel", "reduction"]} ins(%padded_121, %arg33 : tensor<64x9x9xf32>, tensor<64x64x3x3xf32>) outs(%cst_0 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %in_168: f32, %out: f32):
      %310 = arith.mulf %in, %in_168 : f32
      %311 = arith.addf %out, %310 : f32
      linalg.yield %311 : f32
    } -> tensor<64x7x7xf32>
    %expanded_122 = tensor.expand_shape %213 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %expanded_123 = tensor.expand_shape %arg70 [[0, 1, 2, 3]] output_shape [1, 64, 1, 1] : tensor<64xf32> into tensor<1x64x1x1xf32>
    %214 = tensor.empty() : tensor<64xf32>
    %215 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel"]} ins(%cst_6 : tensor<f32>) outs(%214 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64xf32>
    %expanded_124 = tensor.expand_shape %215 [[0, 1, 2, 3]] output_shape [1, 64, 1, 1] : tensor<64xf32> into tensor<1x64x1x1xf32>
    %216 = arith.addf %expanded_123, %expanded_124 : tensor<1x64x1x1xf32>
    %217 = math.rsqrt %216 : tensor<1x64x1x1xf32>
    %218 = tensor.empty() : tensor<64x7x7xf32>
    %219 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg69 : tensor<64xf32>) outs(%218 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x7x7xf32>
    %expanded_125 = tensor.expand_shape %219 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %220 = arith.subf %expanded_122, %expanded_125 : tensor<1x64x7x7xf32>
    %collapsed_126 = tensor.collapse_shape %217 [[0, 1, 2, 3]] : tensor<1x64x1x1xf32> into tensor<64xf32>
    %221 = tensor.empty() : tensor<64x7x7xf32>
    %222 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_126 : tensor<64xf32>) outs(%221 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x7x7xf32>
    %expanded_127 = tensor.expand_shape %222 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %223 = arith.mulf %220, %expanded_127 : tensor<1x64x7x7xf32>
    %224 = tensor.empty() : tensor<64x7x7xf32>
    %225 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg34 : tensor<64xf32>) outs(%224 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x7x7xf32>
    %expanded_128 = tensor.expand_shape %225 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %226 = arith.mulf %223, %expanded_128 : tensor<1x64x7x7xf32>
    %227 = tensor.empty() : tensor<64x7x7xf32>
    %228 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg35 : tensor<64xf32>) outs(%227 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x7x7xf32>
    %expanded_129 = tensor.expand_shape %228 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %229 = arith.addf %226, %expanded_129 : tensor<1x64x7x7xf32>
    %collapsed_130 = tensor.collapse_shape %194 [[0, 1], [2], [3]] : tensor<1x32x14x14xf32> into tensor<32x14x14xf32>
    %collapsed_131 = tensor.collapse_shape %arg36 [[0], [1, 2, 3]] : tensor<64x32x1x1xf32> into tensor<64x32xf32>
    %230 = linalg.generic {indexing_maps = [#map12, #map13, #map14], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%collapsed_130, %collapsed_131 : tensor<32x14x14xf32>, tensor<64x32xf32>) outs(%cst_0 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %in_168: f32, %out: f32):
      %310 = arith.mulf %in, %in_168 : f32
      %311 = arith.addf %out, %310 : f32
      linalg.yield %311 : f32
    } -> tensor<64x7x7xf32>
    %expanded_132 = tensor.expand_shape %230 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %expanded_133 = tensor.expand_shape %arg72 [[0, 1, 2, 3]] output_shape [1, 64, 1, 1] : tensor<64xf32> into tensor<1x64x1x1xf32>
    %231 = tensor.empty() : tensor<64xf32>
    %232 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel"]} ins(%cst_6 : tensor<f32>) outs(%231 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64xf32>
    %expanded_134 = tensor.expand_shape %232 [[0, 1, 2, 3]] output_shape [1, 64, 1, 1] : tensor<64xf32> into tensor<1x64x1x1xf32>
    %233 = arith.addf %expanded_133, %expanded_134 : tensor<1x64x1x1xf32>
    %234 = math.rsqrt %233 : tensor<1x64x1x1xf32>
    %235 = tensor.empty() : tensor<64x7x7xf32>
    %236 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg71 : tensor<64xf32>) outs(%235 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x7x7xf32>
    %expanded_135 = tensor.expand_shape %236 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %237 = arith.subf %expanded_132, %expanded_135 : tensor<1x64x7x7xf32>
    %collapsed_136 = tensor.collapse_shape %234 [[0, 1, 2, 3]] : tensor<1x64x1x1xf32> into tensor<64xf32>
    %238 = tensor.empty() : tensor<64x7x7xf32>
    %239 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_136 : tensor<64xf32>) outs(%238 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x7x7xf32>
    %expanded_137 = tensor.expand_shape %239 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %240 = arith.mulf %237, %expanded_137 : tensor<1x64x7x7xf32>
    %241 = tensor.empty() : tensor<64x7x7xf32>
    %242 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg37 : tensor<64xf32>) outs(%241 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x7x7xf32>
    %expanded_138 = tensor.expand_shape %242 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %243 = arith.mulf %240, %expanded_138 : tensor<1x64x7x7xf32>
    %244 = tensor.empty() : tensor<64x7x7xf32>
    %245 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg38 : tensor<64xf32>) outs(%244 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x7x7xf32>
    %expanded_139 = tensor.expand_shape %245 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %246 = arith.addf %243, %expanded_139 : tensor<1x64x7x7xf32>
    %247 = tensor.empty() : tensor<64x7x7xf32>
    %248 = linalg.generic {indexing_maps = [#map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_5 : tensor<f32>) outs(%247 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x7x7xf32>
    %expanded_140 = tensor.expand_shape %248 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %249 = arith.mulf %246, %expanded_140 : tensor<1x64x7x7xf32>
    %250 = arith.addf %229, %249 : tensor<1x64x7x7xf32>
    %251 = call @relu_33(%250) : (tensor<1x64x7x7xf32>) -> tensor<1x64x7x7xf32>
    %collapsed_141 = tensor.collapse_shape %251 [[0, 1], [2], [3]] : tensor<1x64x7x7xf32> into tensor<64x7x7xf32>
    %padded_142 = tensor.pad %collapsed_141 low[0, 1, 1] high[0, 1, 1] {
    ^bb0(%arg78: index, %arg79: index, %arg80: index):
      tensor.yield %cst_7 : f32
    } : tensor<64x7x7xf32> to tensor<64x9x9xf32>
    %252 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel", "reduction"]} ins(%padded_142, %arg39 : tensor<64x9x9xf32>, tensor<64x64x3x3xf32>) outs(%cst_0 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %in_168: f32, %out: f32):
      %310 = arith.mulf %in, %in_168 : f32
      %311 = arith.addf %out, %310 : f32
      linalg.yield %311 : f32
    } -> tensor<64x7x7xf32>
    %expanded_143 = tensor.expand_shape %252 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %expanded_144 = tensor.expand_shape %arg74 [[0, 1, 2, 3]] output_shape [1, 64, 1, 1] : tensor<64xf32> into tensor<1x64x1x1xf32>
    %253 = tensor.empty() : tensor<64xf32>
    %254 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel"]} ins(%cst_6 : tensor<f32>) outs(%253 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64xf32>
    %expanded_145 = tensor.expand_shape %254 [[0, 1, 2, 3]] output_shape [1, 64, 1, 1] : tensor<64xf32> into tensor<1x64x1x1xf32>
    %255 = arith.addf %expanded_144, %expanded_145 : tensor<1x64x1x1xf32>
    %256 = math.rsqrt %255 : tensor<1x64x1x1xf32>
    %257 = tensor.empty() : tensor<64x7x7xf32>
    %258 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg73 : tensor<64xf32>) outs(%257 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x7x7xf32>
    %expanded_146 = tensor.expand_shape %258 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %259 = arith.subf %expanded_143, %expanded_146 : tensor<1x64x7x7xf32>
    %collapsed_147 = tensor.collapse_shape %256 [[0, 1, 2, 3]] : tensor<1x64x1x1xf32> into tensor<64xf32>
    %260 = tensor.empty() : tensor<64x7x7xf32>
    %261 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_147 : tensor<64xf32>) outs(%260 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x7x7xf32>
    %expanded_148 = tensor.expand_shape %261 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %262 = arith.mulf %259, %expanded_148 : tensor<1x64x7x7xf32>
    %263 = tensor.empty() : tensor<64x7x7xf32>
    %264 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg40 : tensor<64xf32>) outs(%263 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x7x7xf32>
    %expanded_149 = tensor.expand_shape %264 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %265 = arith.mulf %262, %expanded_149 : tensor<1x64x7x7xf32>
    %266 = tensor.empty() : tensor<64x7x7xf32>
    %267 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg41 : tensor<64xf32>) outs(%266 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x7x7xf32>
    %expanded_150 = tensor.expand_shape %267 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %268 = arith.addf %265, %expanded_150 : tensor<1x64x7x7xf32>
    %269 = call @relu_33(%268) : (tensor<1x64x7x7xf32>) -> tensor<1x64x7x7xf32>
    %collapsed_151 = tensor.collapse_shape %269 [[0, 1], [2], [3]] : tensor<1x64x7x7xf32> into tensor<64x7x7xf32>
    %padded_152 = tensor.pad %collapsed_151 low[0, 1, 1] high[0, 1, 1] {
    ^bb0(%arg78: index, %arg79: index, %arg80: index):
      tensor.yield %cst_7 : f32
    } : tensor<64x7x7xf32> to tensor<64x9x9xf32>
    %270 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel", "reduction"]} ins(%padded_152, %arg42 : tensor<64x9x9xf32>, tensor<64x64x3x3xf32>) outs(%cst_0 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %in_168: f32, %out: f32):
      %310 = arith.mulf %in, %in_168 : f32
      %311 = arith.addf %out, %310 : f32
      linalg.yield %311 : f32
    } -> tensor<64x7x7xf32>
    %expanded_153 = tensor.expand_shape %270 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %expanded_154 = tensor.expand_shape %arg76 [[0, 1, 2, 3]] output_shape [1, 64, 1, 1] : tensor<64xf32> into tensor<1x64x1x1xf32>
    %271 = tensor.empty() : tensor<64xf32>
    %272 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel"]} ins(%cst_6 : tensor<f32>) outs(%271 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64xf32>
    %expanded_155 = tensor.expand_shape %272 [[0, 1, 2, 3]] output_shape [1, 64, 1, 1] : tensor<64xf32> into tensor<1x64x1x1xf32>
    %273 = arith.addf %expanded_154, %expanded_155 : tensor<1x64x1x1xf32>
    %274 = math.rsqrt %273 : tensor<1x64x1x1xf32>
    %275 = tensor.empty() : tensor<64x7x7xf32>
    %276 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg75 : tensor<64xf32>) outs(%275 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x7x7xf32>
    %expanded_156 = tensor.expand_shape %276 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %277 = arith.subf %expanded_153, %expanded_156 : tensor<1x64x7x7xf32>
    %collapsed_157 = tensor.collapse_shape %274 [[0, 1, 2, 3]] : tensor<1x64x1x1xf32> into tensor<64xf32>
    %278 = tensor.empty() : tensor<64x7x7xf32>
    %279 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_157 : tensor<64xf32>) outs(%278 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x7x7xf32>
    %expanded_158 = tensor.expand_shape %279 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %280 = arith.mulf %277, %expanded_158 : tensor<1x64x7x7xf32>
    %281 = tensor.empty() : tensor<64x7x7xf32>
    %282 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg43 : tensor<64xf32>) outs(%281 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x7x7xf32>
    %expanded_159 = tensor.expand_shape %282 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %283 = arith.mulf %280, %expanded_159 : tensor<1x64x7x7xf32>
    %284 = tensor.empty() : tensor<64x7x7xf32>
    %285 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg44 : tensor<64xf32>) outs(%284 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x7x7xf32>
    %expanded_160 = tensor.expand_shape %285 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %286 = arith.addf %283, %expanded_160 : tensor<1x64x7x7xf32>
    %287 = tensor.empty() : tensor<64x7x7xf32>
    %288 = linalg.generic {indexing_maps = [#map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_5 : tensor<f32>) outs(%287 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x7x7xf32>
    %expanded_161 = tensor.expand_shape %288 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %289 = arith.mulf %251, %expanded_161 : tensor<1x64x7x7xf32>
    %290 = arith.addf %286, %289 : tensor<1x64x7x7xf32>
    %291 = call @relu_33(%290) : (tensor<1x64x7x7xf32>) -> tensor<1x64x7x7xf32>
    %collapsed_162 = tensor.collapse_shape %291 [[0, 1], [2], [3]] : tensor<1x64x7x7xf32> into tensor<64x7x7xf32>
    %292 = linalg.generic {indexing_maps = [#map15, #map5], iterator_types = ["parallel", "reduction", "reduction"]} ins(%collapsed_162 : tensor<64x7x7xf32>) outs(%cst : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %310 = arith.addf %out, %in : f32
      linalg.yield %310 : f32
    } -> tensor<64xf32>
    %293 = tensor.empty() : tensor<64xf32>
    %294 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%292 : tensor<64xf32>) outs(%293 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64xf32>
    %expanded_163 = tensor.expand_shape %294 [[0, 1, 2, 3]] output_shape [1, 64, 1, 1] : tensor<64xf32> into tensor<1x64x1x1xf32>
    %295 = tensor.empty() : tensor<64xf32>
    %296 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel"]} ins(%cst_4 : tensor<f32>) outs(%295 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64xf32>
    %expanded_164 = tensor.expand_shape %296 [[0, 1, 2, 3]] output_shape [1, 64, 1, 1] : tensor<64xf32> into tensor<1x64x1x1xf32>
    %297 = arith.divf %expanded_163, %expanded_164 : tensor<1x64x1x1xf32>
    %collapsed_165 = tensor.collapse_shape %297 [[0], [1, 2, 3]] : tensor<1x64x1x1xf32> into tensor<1x64xf32>
    %298 = tensor.empty() : tensor<64x10xf32>
    %299 = linalg.generic {indexing_maps = [#map16, #map17], iterator_types = ["parallel", "parallel"]} ins(%arg45 : tensor<10x64xf32>) outs(%298 : tensor<64x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x10xf32>
    %300 = tensor.empty() : tensor<10xf32>
    %301 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel"]} ins(%cst_5 : tensor<f32>) outs(%300 : tensor<10xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<10xf32>
    %302 = arith.mulf %arg46, %301 : tensor<10xf32>
    %303 = linalg.matmul ins(%collapsed_165, %299 : tensor<1x64xf32>, tensor<64x10xf32>) outs(%cst_3 : tensor<1x10xf32>) -> tensor<1x10xf32>
    %304 = tensor.empty() : tensor<10xf32>
    %305 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel"]} ins(%cst_5 : tensor<f32>) outs(%304 : tensor<10xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<10xf32>
    %expanded_166 = tensor.expand_shape %305 [[0, 1]] output_shape [1, 10] : tensor<10xf32> into tensor<1x10xf32>
    %306 = arith.mulf %expanded_166, %303 : tensor<1x10xf32>
    %307 = tensor.empty() : tensor<10xf32>
    %308 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%302 : tensor<10xf32>) outs(%307 : tensor<10xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<10xf32>
    %expanded_167 = tensor.expand_shape %308 [[0, 1]] output_shape [1, 10] : tensor<10xf32> into tensor<1x10xf32>
    %309 = arith.addf %expanded_167, %306 : tensor<1x10xf32>
    return %309 : tensor<1x10xf32>
  }
  func.func private @relu(%arg0: tensor<1x16x28x28xf32>) -> tensor<1x16x28x28xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %0 = tensor.empty() : tensor<16x28x28xf32>
    %1 = linalg.generic {indexing_maps = [#map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : tensor<f32>) outs(%0 : tensor<16x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x28x28xf32>
    %expanded = tensor.expand_shape %1 [[0, 1], [2], [3]] output_shape [1, 16, 28, 28] : tensor<16x28x28xf32> into tensor<1x16x28x28xf32>
    %2 = arith.maximumf %arg0, %expanded : tensor<1x16x28x28xf32>
    return %2 : tensor<1x16x28x28xf32>
  }
  func.func private @relu_16(%arg0: tensor<1x32x14x14xf32>) -> tensor<1x32x14x14xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %0 = tensor.empty() : tensor<32x14x14xf32>
    %1 = linalg.generic {indexing_maps = [#map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : tensor<f32>) outs(%0 : tensor<32x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x14x14xf32>
    %expanded = tensor.expand_shape %1 [[0, 1], [2], [3]] output_shape [1, 32, 14, 14] : tensor<32x14x14xf32> into tensor<1x32x14x14xf32>
    %2 = arith.maximumf %arg0, %expanded : tensor<1x32x14x14xf32>
    return %2 : tensor<1x32x14x14xf32>
  }
  func.func private @relu_33(%arg0: tensor<1x64x7x7xf32>) -> tensor<1x64x7x7xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %0 = tensor.empty() : tensor<64x7x7xf32>
    %1 = linalg.generic {indexing_maps = [#map10, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : tensor<f32>) outs(%0 : tensor<64x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x7x7xf32>
    %expanded = tensor.expand_shape %1 [[0, 1], [2], [3]] output_shape [1, 64, 7, 7] : tensor<64x7x7xf32> into tensor<1x64x7x7xf32>
    %2 = arith.maximumf %arg0, %expanded : tensor<1x64x7x7xf32>
    return %2 : tensor<1x64x7x7xf32>
  }
}
