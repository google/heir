// This file contains a minimal torch-exported (and canonicalized) mnist model
// composed of feedforward layers and ReLU activation. The export path is torch
// -> stablehlo -> mlir.

module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @mnist(%arg0: tensor<512x784xf32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<512xf32> {mhlo.sharding = "{replicated}"}, %arg2: tensor<10x512xf32> {mhlo.sharding = "{replicated}"}, %arg3: tensor<10xf32> {mhlo.sharding = "{replicated}"}, %arg4: tensor<1x784xf32> {secret.secret}) -> (tensor<1x10xf32> {jax.result_info = "result[0]"}) {
    %cst = arith.constant dense<1.000000e+00> : tensor<f32>
    %0 = tensor.empty() : tensor<784x512xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<512x784xf32>) outs(%0 : tensor<784x512xf32>) permutation = [1, 0]
    %1 = tensor.empty() : tensor<512xf32>
    %broadcasted = linalg.broadcast ins(%cst : tensor<f32>) outs(%1 : tensor<512xf32>) dimensions = [0]
    %2 = tensor.empty() : tensor<512xf32>
    %mapped = linalg.map { arith.mulf } ins(%arg1, %broadcasted : tensor<512xf32>, tensor<512xf32>) outs(%2 : tensor<512xf32>)
    %3 = tensor.empty() : tensor<1x512xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %4 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %5 = linalg.matmul ins(%arg4, %transposed : tensor<1x784xf32>, tensor<784x512xf32>) outs(%4 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %6 = tensor.empty() : tensor<1x512xf32>
    %broadcasted_1 = linalg.broadcast ins(%cst : tensor<f32>) outs(%6 : tensor<1x512xf32>) dimensions = [0, 1]
    %7 = tensor.empty() : tensor<1x512xf32>
    %mapped_2 = linalg.map { arith.mulf } ins(%broadcasted_1, %5 : tensor<1x512xf32>, tensor<1x512xf32>) outs(%7 : tensor<1x512xf32>)
    %8 = tensor.empty() : tensor<1x512xf32>
    %broadcasted_3 = linalg.broadcast ins(%mapped : tensor<512xf32>) outs(%8 : tensor<1x512xf32>) dimensions = [0]
    %9 = tensor.empty() : tensor<1x512xf32>
    %mapped_4 = linalg.map { arith.addf } ins(%broadcasted_3, %mapped_2 : tensor<1x512xf32>, tensor<1x512xf32>) outs(%9 : tensor<1x512xf32>)
    %10 = call @relu(%mapped_4) { domain_lower = -15.0, domain_upper = 12.0 } : (tensor<1x512xf32>) -> tensor<1x512xf32>
    %11 = tensor.empty() : tensor<512x10xf32>
    %transposed_5 = linalg.transpose ins(%arg2 : tensor<10x512xf32>) outs(%11 : tensor<512x10xf32>) permutation = [1, 0]
    %12 = tensor.empty() : tensor<10xf32>
    %broadcasted_6 = linalg.broadcast ins(%cst : tensor<f32>) outs(%12 : tensor<10xf32>) dimensions = [0]
    %13 = tensor.empty() : tensor<10xf32>
    %mapped_7 = linalg.map { arith.mulf } ins(%arg3, %broadcasted_6 : tensor<10xf32>, tensor<10xf32>) outs(%13 : tensor<10xf32>)
    %14 = tensor.empty() : tensor<1x10xf32>
    %cst_8 = arith.constant 0.000000e+00 : f32
    %15 = linalg.fill ins(%cst_8 : f32) outs(%14 : tensor<1x10xf32>) -> tensor<1x10xf32>
    %16 = linalg.matmul ins(%10, %transposed_5 : tensor<1x512xf32>, tensor<512x10xf32>) outs(%15 : tensor<1x10xf32>) -> tensor<1x10xf32>
    %17 = tensor.empty() : tensor<1x10xf32>
    %broadcasted_9 = linalg.broadcast ins(%cst : tensor<f32>) outs(%17 : tensor<1x10xf32>) dimensions = [0, 1]
    %18 = tensor.empty() : tensor<1x10xf32>
    %mapped_10 = linalg.map { arith.mulf } ins(%broadcasted_9, %16 : tensor<1x10xf32>, tensor<1x10xf32>) outs(%18 : tensor<1x10xf32>)
    %19 = tensor.empty() : tensor<1x10xf32>
    %broadcasted_11 = linalg.broadcast ins(%mapped_7 : tensor<10xf32>) outs(%19 : tensor<1x10xf32>) dimensions = [0]
    %20 = tensor.empty() : tensor<1x10xf32>
    %mapped_12 = linalg.map { arith.addf } ins(%broadcasted_11, %mapped_10 : tensor<1x10xf32>, tensor<1x10xf32>) outs(%20 : tensor<1x10xf32>)
    return %mapped_12 : tensor<1x10xf32>
  }
  func.func private @relu(%arg0: tensor<1x512xf32>) -> tensor<1x512xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %0 = tensor.empty() : tensor<1x512xf32>
    %broadcasted = linalg.broadcast ins(%cst : tensor<f32>) outs(%0 : tensor<1x512xf32>) dimensions = [0, 1]
    %1 = tensor.empty() : tensor<1x512xf32>
    %mapped = linalg.map { arith.maximumf } ins(%arg0, %broadcasted : tensor<1x512xf32>, tensor<1x512xf32>) outs(%1 : tensor<1x512xf32>)
    return %mapped : tensor<1x512xf32>
  }
}
