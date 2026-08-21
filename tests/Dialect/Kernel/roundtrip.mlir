// RUN: heir-opt %s | FileCheck %s

#key = #lwe.key<slot_index = 0>
!Z65537_i64 = !mod_arith.int<65537 : i64>
#ring_Z65537_i64_1_x1024 = #polynomial.ring<coefficientType = !Z65537_i64, polynomialModulus = <1 + x**1024>>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024, encoding = #full_crt_packing_encoding>
!Z1095233372161_i64 = !mod_arith.int<1095233372161 : i64>
!rns_L0 = !rns.rns<!Z1095233372161_i64>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = lsb>
!ciphertext_rlwe = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0, key = #key>

// CHECK: module
module {
  // CHECK: @test_chebyshev
  func.func @test_chebyshev(%arg0: f64) -> f64 {
    // CHECK: kernel.eval_chebyshev %arg0 {coefficients = [1.000000e+00, 2.000000e+00]} : f64 -> f64
    %0 = kernel.eval_chebyshev %arg0 {coefficients = [1.0, 2.0]} : f64 -> f64
    return %0 : f64
  }

  // CHECK: @test_linear_transform_tensor_1d
  func.func @test_linear_transform_tensor_1d(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: kernel.linear_transform %arg0, %{{.*}} {diagonal_indices = array<i64: 0, 1>} : tensor<4xf32>, tensor<2x4xf32> -> tensor<4xf32>
    %diagonals = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
    %0 = kernel.linear_transform %arg0, %diagonals {
      diagonal_indices = array<i64: 0, 1>
    } : tensor<4xf32>, tensor<2x4xf32> -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK: @test_linear_transform_tensor_2d
  func.func @test_linear_transform_tensor_2d(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
    // CHECK: kernel.linear_transform %arg0, %{{.*}} {diagonal_indices = array<i64: 0, 1>} : tensor<1x4xf32>, tensor<2x4xf32> -> tensor<1x4xf32>
    %diagonals = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
    %0 = kernel.linear_transform %arg0, %diagonals {
      diagonal_indices = array<i64: 0, 1>
    } : tensor<1x4xf32>, tensor<2x4xf32> -> tensor<1x4xf32>
    return %0 : tensor<1x4xf32>
  }

  // CHECK: @test_linear_transform_bsgs
  func.func @test_linear_transform_bsgs(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: kernel.linear_transform %arg0, %{{.*}} {bsgs_ratio = 5.000000e-01 : f64, diagonal_indices = array<i64: 0, 1>} : tensor<4xf32>, tensor<2x4xf32> -> tensor<4xf32>
    %diagonals = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
    %0 = kernel.linear_transform %arg0, %diagonals {
      diagonal_indices = array<i64: 0, 1>,
      bsgs_ratio = 0.5 : f64
    } : tensor<4xf32>, tensor<2x4xf32> -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK: @test_linear_transform_lwe
  // CHECK: kernel.linear_transform %arg0, %{{.*}} {diagonal_indices = array<i64: 0, 1>} : tensor<1x!ct_L0>, tensor<2x1024xf64> -> tensor<1x!ct_L0>
  func.func @test_linear_transform_lwe(%arg0: tensor<1x!ciphertext_rlwe>) -> tensor<1x!ciphertext_rlwe> {
    %diagonals = arith.constant dense<1.0> : tensor<2x1024xf64>
    %0 = kernel.linear_transform %arg0, %diagonals {
      diagonal_indices = array<i64: 0, 1>
    } : tensor<1x!ciphertext_rlwe>, tensor<2x1024xf64> -> tensor<1x!ciphertext_rlwe>
    return %0 : tensor<1x!ciphertext_rlwe>
  }

  // CHECK: @test_prepare_apply
  func.func @test_prepare_apply(%arg0: tensor<1x!ciphertext_rlwe>) -> tensor<1x!ciphertext_rlwe> {
    %diagonals = arith.constant dense<1.0> : tensor<2x1024xf64>
    // CHECK: kernel.prepare_linear_transform %{{.*}} {diagonal_indices = array<i64: 0, 1>} : tensor<2x1024xf64> -> <level = 0, slots = 1024, log_bsgs_ratio = 0>
    %lt = kernel.prepare_linear_transform %diagonals {
      diagonal_indices = array<i64: 0, 1>
    } : tensor<2x1024xf64> -> !kernel.prepared_linear_transform<level = 0, slots = 1024, log_bsgs_ratio = 0>
    // CHECK: kernel.apply_linear_transform %arg0, %{{.*}} : tensor<1x!ct_L0>, <level = 0, slots = 1024, log_bsgs_ratio = 0> -> tensor<1x!ct_L0>
    %0 = kernel.apply_linear_transform %arg0, %lt : tensor<1x!ciphertext_rlwe>, !kernel.prepared_linear_transform<level = 0, slots = 1024, log_bsgs_ratio = 0> -> tensor<1x!ciphertext_rlwe>
    return %0 : tensor<1x!ciphertext_rlwe>
  }
}
