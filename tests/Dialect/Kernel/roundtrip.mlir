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

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#ckks_key = #lwe.key<>
#ckks_modulus_chain = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_ckks_L0 = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
#ring_rns_ckks_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_ckks_L0, polynomialModulus = <1 + x**1024>>
#ckks_ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_ckks_L0_1_x1024, encryption_type = mix>
!ckks_ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ckks_ciphertext_space_L0, key = #ckks_key, modulus_chain = #ckks_modulus_chain>

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

  // CHECK: @test_linear_transform_integer
  func.func @test_linear_transform_integer(%arg0: tensor<4xi32>) -> tensor<4xi32> {
    // CHECK: kernel.linear_transform %arg0, %{{.*}} {diagonal_indices = array<i64: 0, 1>} : tensor<4xi32>, tensor<2x4xi32> -> tensor<4xi32>
    %diagonals = arith.constant dense<[[1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi32>
    %0 = kernel.linear_transform %arg0, %diagonals {
      diagonal_indices = array<i64: 0, 1>
    } : tensor<4xi32>, tensor<2x4xi32> -> tensor<4xi32>
    return %0 : tensor<4xi32>
  }

  // A single ciphertext, as produced by --convert-elementwise-to-affine.
  // CHECK: @test_linear_transform_scalar_ciphertext
  // CHECK: kernel.linear_transform %{{.*}}, %{{.*}} {diagonal_indices = array<i64: 0, 1>} : !ct_L0, tensor<2x1024xf64> -> !ct_L0
  func.func @test_linear_transform_scalar_ciphertext(%arg0: !ciphertext_rlwe) -> !ciphertext_rlwe {
    %diagonals = arith.constant dense<1.0> : tensor<2x1024xf64>
    %0 = kernel.linear_transform %arg0, %diagonals {
      diagonal_indices = array<i64: 0, 1>
    } : !ciphertext_rlwe, tensor<2x1024xf64> -> !ciphertext_rlwe
    return %0 : !ciphertext_rlwe
  }

  // CHECK: @test_prepare_integer
  func.func @test_prepare_integer() -> !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0> {
    %diagonals = arith.constant dense<[[1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi64>
    // CHECK: kernel.prepare_linear_transform %{{.*}} {diagonal_indices = array<i64: 0, 1>} : tensor<2x4xi64> -> <level = 0, slots = 4, log_bsgs_ratio = 0>
    %lt = kernel.prepare_linear_transform %diagonals {
      diagonal_indices = array<i64: 0, 1>
    } : tensor<2x4xi64> -> !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>
    return %lt : !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>
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

  // Diagonals exactly one ciphertext wide (512 CKKS slots) are valid for a
  // tensor of ciphertexts, which are transformed independently.
  // CHECK: @test_linear_transform_multi_ciphertext
  // CHECK: kernel.linear_transform %arg0, %{{.*}} {diagonal_indices = array<i64: 0, 1>} : tensor<2x!{{.*}}>, tensor<2x512xf64> -> tensor<2x!{{.*}}>
  func.func @test_linear_transform_multi_ciphertext(%arg0: tensor<2x!ckks_ct>) -> tensor<2x!ckks_ct> {
    %diagonals = arith.constant dense<1.0> : tensor<2x512xf64>
    %0 = kernel.linear_transform %arg0, %diagonals {
      diagonal_indices = array<i64: 0, 1>
    } : tensor<2x!ckks_ct>, tensor<2x512xf64> -> tensor<2x!ckks_ct>
    return %0 : tensor<2x!ckks_ct>
  }

  // CHECK: @test_linear_transform_scalar_ckks_ciphertext
  // CHECK: kernel.linear_transform %{{.*}}, %{{.*}} {diagonal_indices = array<i64: 0, 1>} : !{{.*}}, tensor<2x512xf64> -> !{{.*}}
  func.func @test_linear_transform_scalar_ckks_ciphertext(%arg0: !ckks_ct) -> !ckks_ct {
    %diagonals = arith.constant dense<1.0> : tensor<2x512xf64>
    %0 = kernel.linear_transform %arg0, %diagonals {
      diagonal_indices = array<i64: 0, 1>
    } : !ckks_ct, tensor<2x512xf64> -> !ckks_ct
    return %0 : !ckks_ct
  }

  // A prepared transform exactly one ciphertext wide (512 CKKS slots) applies
  // to each ciphertext of a tensor, and to a scalar ciphertext.
  // CHECK: @test_apply_multi_ciphertext
  func.func @test_apply_multi_ciphertext(%arg0: tensor<2x!ckks_ct>, %arg1: !ckks_ct) -> (tensor<2x!ckks_ct>, !ckks_ct) {
    %diagonals = arith.constant dense<1.0> : tensor<2x512xf64>
    // CHECK: kernel.prepare_linear_transform %{{.*}} {diagonal_indices = array<i64: 0, 1>} : tensor<2x512xf64> -> <level = 0, slots = 512, log_bsgs_ratio = 0>
    %lt = kernel.prepare_linear_transform %diagonals {
      diagonal_indices = array<i64: 0, 1>
    } : tensor<2x512xf64> -> !kernel.prepared_linear_transform<level = 0, slots = 512, log_bsgs_ratio = 0>
    // CHECK: kernel.apply_linear_transform %arg0, %{{.*}} : tensor<2x!{{.*}}>, <level = 0, slots = 512, log_bsgs_ratio = 0> -> tensor<2x!{{.*}}>
    %0 = kernel.apply_linear_transform %arg0, %lt : tensor<2x!ckks_ct>, !kernel.prepared_linear_transform<level = 0, slots = 512, log_bsgs_ratio = 0> -> tensor<2x!ckks_ct>
    // CHECK: kernel.apply_linear_transform %{{.*}}, %{{.*}} : !{{.*}}, <level = 0, slots = 512, log_bsgs_ratio = 0> -> !{{.*}}
    %1 = kernel.apply_linear_transform %arg1, %lt : !ckks_ct, !kernel.prepared_linear_transform<level = 0, slots = 512, log_bsgs_ratio = 0> -> !ckks_ct
    return %0, %1 : tensor<2x!ckks_ct>, !ckks_ct
  }
}
