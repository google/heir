// This file contains a minimal torch-exported (and canonicalized) mnist model
// composed of feedforward layers and ReLU activation. The export path is torch
// -> stablehlo -> mlir.

module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @mnist(%arg0: tensor<3x5xf32> {mhlo.sharding = "{replicated}"}, %arg4: tensor<1x5xf32> {secret.secret}) -> (tensor<1x3xf32> {jax.result_info = "result[0]"}) {
    %0 = tensor.empty() : tensor<5x3xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<3x5xf32>) outs(%0 : tensor<5x3xf32>) permutation = [1, 0]
    %3 = tensor.empty() : tensor<1x3xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %4 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %5 = linalg.matmul ins(%arg4, %transposed : tensor<1x5xf32>, tensor<5x3xf32>) outs(%4 : tensor<1x3xf32>) -> tensor<1x3xf32>
    return %5 : tensor<1x3xf32>
  }
}
