// RUN: heir-opt --inline-activations --activation-canonicalizations --linalg-canonicalizations --linalg-fuse-linear-ops --canonicalize %s | FileCheck %s

// This test verifies that the preprocessing + canonicalization + fusion
// pipeline can successfully fold and fuse a raw BatchNorm sequence exported
// from torch-mlir when using standard `dense` constants.

// CHECK: func.func @reproducer_raw
// CHECK: arith.constant
// CHECK-NEXT: arith.constant
// CHECK-NEXT: linalg.matmul
// CHECK-NEXT: return
func.func @reproducer_raw(%arg0: tensor<1x8xf32>) -> tensor<1x16xf32> {
  %cst_w = arith.constant dense<2.000000e+00> : tensor<16x8xf32> // Use 2.0 to prevent folding to identity
  %cst_b = arith.constant dense<2.000000e+00> : tensor<16xf32>

  // BatchNorm parameters
  %cst_mean = arith.constant dense<3.000000e+00> : tensor<16xf32>
  %cst_var = arith.constant dense<4.000000e+00> : tensor<16xf32>
  %cst_gamma = arith.constant dense<5.000000e+00> : tensor<16xf32>
  %cst_beta = arith.constant dense<6.000000e+00> : tensor<16xf32>
  %cst_eps = arith.constant 1.000000e-05 : f32

  %cst_zero = arith.constant dense<0.000000e+00> : tensor<1x16xf32>
  %cst_one = arith.constant 1.000000e+00 : f32

  %0 = tensor.empty() : tensor<8x16xf32>
  %transposed = linalg.transpose ins(%cst_w : tensor<16x8xf32>) outs(%0 : tensor<8x16xf32>) permutation = [1, 0]

  // MatMul
  %1 = linalg.matmul ins(%arg0, %transposed : tensor<1x8xf32>, tensor<8x16xf32>) outs(%cst_zero : tensor<1x16xf32>) -> tensor<1x16xf32>

  // Bias Add
  %expanded_b = tensor.expand_shape %cst_b [[0, 1]] output_shape [1, 16] : tensor<16xf32> into tensor<1x16xf32>
  %2 = arith.addf %1, %expanded_b : tensor<1x16xf32>

  // BatchNorm Step 1: Compute reciprocal standard deviation: 1 / sqrt(var + eps)
  %empty_16 = tensor.empty() : tensor<16xf32>
  %eps_filled = linalg.fill ins(%cst_eps : f32) outs(%empty_16 : tensor<16xf32>) -> tensor<16xf32>
  %var_plus_eps = arith.addf %cst_var, %eps_filled : tensor<16xf32>

  %std = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%var_plus_eps : tensor<16xf32>) outs(%empty_16 : tensor<16xf32>) {
  ^bb0(%in: f32, %out: f32):
    %sqrt = math.sqrt %in : f32
    linalg.yield %sqrt : f32
  } -> tensor<16xf32>

  %reciprocal_std = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%std : tensor<16xf32>) outs(%empty_16 : tensor<16xf32>) {
  ^bb0(%in: f32, %out: f32):
    %div = arith.divf %cst_one, %in : f32
    linalg.yield %div : f32
  } -> tensor<16xf32>

  // BatchNorm Step 2: Subtract Mean
  %expanded_mean = tensor.expand_shape %cst_mean [[0, 1]] output_shape [1, 16] : tensor<16xf32> into tensor<1x16xf32>
  %3 = arith.subf %2, %expanded_mean : tensor<1x16xf32>

  // BatchNorm Step 3: Multiply by reciprocal std
  %expanded_rstd = tensor.expand_shape %reciprocal_std [[0, 1]] output_shape [1, 16] : tensor<16xf32> into tensor<1x16xf32>
  %4 = arith.mulf %3, %expanded_rstd : tensor<1x16xf32>

  // BatchNorm Step 4: Multiply by gamma
  %expanded_gamma = tensor.expand_shape %cst_gamma [[0, 1]] output_shape [1, 16] : tensor<16xf32> into tensor<1x16xf32>
  %5 = arith.mulf %4, %expanded_gamma : tensor<1x16xf32>

  // BatchNorm Step 5: Add beta
  %expanded_beta = tensor.expand_shape %cst_beta [[0, 1]] output_shape [1, 16] : tensor<16xf32> into tensor<1x16xf32>
  %6 = arith.addf %5, %expanded_beta : tensor<1x16xf32>

  return %6 : tensor<1x16xf32>
}
