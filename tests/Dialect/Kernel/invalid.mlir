// RUN: heir-opt --split-input-file --verify-diagnostics %s

func.func @test_diagonals_not_tensor_parse(%arg0: tensor<4xf32>, %diagonals: f32) -> tensor<4xf32> {
  // The typed custom parser rejects a non-tensor type before verification.
  // expected-error@+1 {{invalid kind of type specified}}
  %0 = kernel.linear_transform %arg0, %diagonals {diagonal_indices = array<i64: 0>} : tensor<4xf32>, f32 -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @test_diagonals_not_shaped_generic(%arg0: tensor<4xf32>, %diagonals: f32) -> tensor<4xf32> {
  // expected-error@below {{operand #1 must be 2D tensor of floating-point or integer values}}
  %0 = "kernel.linear_transform"(%arg0, %diagonals) <{diagonal_indices = array<i64: 0>}> : (tensor<4xf32>, f32) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @test_diagonals_not_2d(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %diagonals = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  // expected-error@below {{operand #1 must be 2D tensor of floating-point or integer values}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0>
  } : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @test_diagonals_3d(%arg0: tensor<4xf32>, %diagonals: tensor<1x2x4xf32>) -> tensor<4xf32> {
  // expected-error@below {{operand #1 must be 2D tensor of floating-point or integer values}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : tensor<4xf32>, tensor<1x2x4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @test_diagonals_complex(%arg0: tensor<4xf32>, %diagonals: tensor<2x4xcomplex<f32>>) -> tensor<4xf32> {
  // expected-error@below {{operand #1 must be 2D tensor of floating-point or integer values}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : tensor<4xf32>, tensor<2x4xcomplex<f32>> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @test_diagonals_index(%arg0: tensor<4xf32>, %diagonals: tensor<2x4xindex>) -> tensor<4xf32> {
  // expected-error@below {{operand #1 must be 2D tensor of floating-point or integer values}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : tensor<4xf32>, tensor<2x4xindex> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @test_input_index(%arg0: tensor<4xindex>, %diagonals: tensor<2x4xf32>) -> tensor<4xindex> {
  // expected-error@below {{operand #0 must be ranked tensor of integer, floating-point, or secret-typed values, or a single secret-typed value}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : tensor<4xindex>, tensor<2x4xf32> -> tensor<4xindex>
  return %0 : tensor<4xindex>
}

// -----

func.func @test_input_cleartext_scalar(%arg0: f32, %diagonals: tensor<2x4xf32>) -> f32 {
  // expected-error@below {{operand #0 must be ranked tensor of integer, floating-point, or secret-typed values, or a single secret-typed value}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : f32, tensor<2x4xf32> -> f32
  return %0 : f32
}

// -----

func.func @test_input_unranked(%arg0: tensor<*xf32>, %diagonals: tensor<2x4xf32>) -> tensor<*xf32> {
  // expected-error@below {{operand #0 must be ranked tensor of integer, floating-point, or secret-typed values, or a single secret-typed value}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : tensor<*xf32>, tensor<2x4xf32> -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func.func @test_input_memref(%arg0: memref<4xf32>, %diagonals: tensor<2x4xf32>) -> memref<4xf32> {
  // expected-error@below {{operand #0 must be ranked tensor of integer, floating-point, or secret-typed values, or a single secret-typed value}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : memref<4xf32>, tensor<2x4xf32> -> memref<4xf32>
  return %0 : memref<4xf32>
}

// -----

func.func @test_input_not_1d_or_2d(%arg0: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
  %diagonals = arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  // expected-error@below {{input must be 1D or 2D ranked tensor}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : tensor<1x2x3xf32>, tensor<2x3xf32> -> tensor<1x2x3xf32>
  return %0 : tensor<1x2x3xf32>
}

// -----

func.func @test_slot_size_mismatch(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %diagonals = arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  // expected-error@below {{input slot size (2) is smaller than diagonals slot size (3)}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : tensor<2xf32>, tensor<2x3xf32> -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

func.func @test_diagonals_indices_mismatch(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %diagonals = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
  // expected-error@below {{number of diagonals (2) must match number of diagonal indices (1)}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0>
  } : tensor<4xf32>, tensor<2x4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @test_batch_dim_not_1(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %diagonals = arith.constant dense<[[1.0, 2.0, 3.0, 4.0]]> : tensor<1x4xf32>
  // expected-error@below {{input tensor batch dimension (first dimension) must be 1}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0>
  } : tensor<2x4xf32>, tensor<1x4xf32> -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @test_prepare_diagonals_not_2d() -> !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0> {
  %diagonals = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  // expected-error@below {{operand #0 must be 2D tensor of floating-point or integer values}}
  %0 = kernel.prepare_linear_transform %diagonals {
    diagonal_indices = array<i64: 0>
  } : tensor<4xf32> -> !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>
  return %0 : !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>
}

// -----

func.func @test_prepare_diagonals_not_tensor_parse(%diagonals: f32) -> !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0> {
  // The typed custom parser rejects a non-tensor type before verification.
  // expected-error@+1 {{invalid kind of type specified}}
  %0 = kernel.prepare_linear_transform %diagonals {diagonal_indices = array<i64: 0>} : f32 -> !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>
  return %0 : !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>
}

// -----

func.func @test_prepare_diagonals_indices_mismatch() -> !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0> {
  %diagonals = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
  // expected-error@below {{number of diagonals (2) must match number of diagonal indices (1)}}
  %0 = kernel.prepare_linear_transform %diagonals {
    diagonal_indices = array<i64: 0>
  } : tensor<2x4xf32> -> !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>
  return %0 : !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>
}

// -----

func.func @test_prepare_source_row_out_of_bounds() -> !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0> {
  %diagonals = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
  // expected-error@below {{source row index 2 is out of bounds for 2 diagonal rows}}
  %0 = kernel.prepare_linear_transform %diagonals {
    diagonal_indices = array<i64: 0>, source_row_indices = array<i64: 2>
  } : tensor<2x4xf32> -> !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>
  return %0 : !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>
}

// -----

func.func @test_prepare_slots_too_small() -> !kernel.prepared_linear_transform<level = 0, slots = 2, log_bsgs_ratio = 0> {
  %diagonals = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
  // expected-error@below {{diagonals slot size (4) exceeds the prepared slot count (2)}}
  %0 = kernel.prepare_linear_transform %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : tensor<2x4xf32> -> !kernel.prepared_linear_transform<level = 0, slots = 2, log_bsgs_ratio = 0>
  return %0 : !kernel.prepared_linear_transform<level = 0, slots = 2, log_bsgs_ratio = 0>
}

// -----

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0 = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = mix>
!ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain>

// The ciphertext sits at level 0 (its chain's current); a transform
// prepared for level 1 must be rejected.
func.func @test_apply_level_mismatch(%ct: !ct) -> !ct {
  %diagonals = arith.constant dense<1.0> : tensor<2x512xf64>
  %lt = kernel.prepare_linear_transform %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : tensor<2x512xf64> -> !kernel.prepared_linear_transform<level = 1, slots = 512, log_bsgs_ratio = 0>
  // expected-error@below {{input ciphertext level (0) does not match the prepared transform level (1)}}
  %0 = kernel.apply_linear_transform %ct, %lt : !ct, !kernel.prepared_linear_transform<level = 1, slots = 512, log_bsgs_ratio = 0> -> !ct
  return %0 : !ct
}

// -----

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>>

// Plaintexts are neither cleartext scalars nor secret-typed.
func.func @test_input_plaintext(%arg0: tensor<1x!pt>) -> tensor<1x!pt> {
  %diagonals = arith.constant dense<1.0> : tensor<2x512xf64>
  // expected-error@below {{operand #0 must be ranked tensor of integer, floating-point, or secret-typed values, or a single secret-typed value}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : tensor<1x!pt>, tensor<2x512xf64> -> tensor<1x!pt>
  return %0 : tensor<1x!pt>
}

// -----

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0 = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = mix>
!ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain>

// Each ciphertext is transformed independently, so the diagonals must fit in
// one ciphertext's 512 slots, not in the 2 * 512 slots of the whole tensor.
func.func @test_multi_ciphertext_diagonals_too_wide(%arg0: tensor<2x!ct>) -> tensor<2x!ct> {
  %diagonals = arith.constant dense<1.0> : tensor<2x1024xf64>
  // expected-error@below {{input slot size (512) is smaller than diagonals slot size (1024)}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : tensor<2x!ct>, tensor<2x1024xf64> -> tensor<2x!ct>
  return %0 : tensor<2x!ct>
}

// -----

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0 = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = mix>
!ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain>

// A scalar ciphertext (e.g. after elementwise-to-affine) is still checked.
func.func @test_scalar_ciphertext_diagonals_too_wide(%arg0: !ct) -> !ct {
  %diagonals = arith.constant dense<1.0> : tensor<2x1024xf64>
  // expected-error@below {{input slot size (512) is smaller than diagonals slot size (1024)}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : !ct, tensor<2x1024xf64> -> !ct
  return %0 : !ct
}

// -----

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0 = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = mix>
!ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain>

// A prepared transform is applied per ciphertext as well.
func.func @test_apply_multi_ciphertext_slots_too_small(%arg0: tensor<2x!ct>) -> tensor<2x!ct> {
  %diagonals = arith.constant dense<1.0> : tensor<2x1024xf64>
  %lt = kernel.prepare_linear_transform %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : tensor<2x1024xf64> -> !kernel.prepared_linear_transform<level = 0, slots = 1024, log_bsgs_ratio = 0>
  // expected-error@below {{input slot size (512) is smaller than the prepared slot count (1024)}}
  %0 = kernel.apply_linear_transform %arg0, %lt : tensor<2x!ct>, !kernel.prepared_linear_transform<level = 0, slots = 1024, log_bsgs_ratio = 0> -> tensor<2x!ct>
  return %0 : tensor<2x!ct>
}

// -----

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0 = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = mix>
!ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain>

func.func @test_apply_scalar_ciphertext_slots_too_small(%arg0: !ct) -> !ct {
  %diagonals = arith.constant dense<1.0> : tensor<2x1024xf64>
  %lt = kernel.prepare_linear_transform %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : tensor<2x1024xf64> -> !kernel.prepared_linear_transform<level = 0, slots = 1024, log_bsgs_ratio = 0>
  // expected-error@below {{input slot size (512) is smaller than the prepared slot count (1024)}}
  %0 = kernel.apply_linear_transform %arg0, %lt : !ct, !kernel.prepared_linear_transform<level = 0, slots = 1024, log_bsgs_ratio = 0> -> !ct
  return %0 : !ct
}

// -----

func.func @test_dynamic_input(%arg0: tensor<?xf64>) -> tensor<?xf64> {
  %diagonals = arith.constant dense<1.0> : tensor<2x4xf64>
  // expected-error@below {{input must have a static shape}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : tensor<?xf64>, tensor<2x4xf64> -> tensor<?xf64>
  return %0 : tensor<?xf64>
}

// -----

func.func @test_dynamic_input_2d(%arg0: tensor<1x?xf64>) -> tensor<1x?xf64> {
  %diagonals = arith.constant dense<1.0> : tensor<2x4xf64>
  // expected-error@below {{input must have a static shape}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : tensor<1x?xf64>, tensor<2x4xf64> -> tensor<1x?xf64>
  return %0 : tensor<1x?xf64>
}

// -----

func.func @test_dynamic_diagonals(%arg0: tensor<4xf64>, %diagonals: tensor<2x?xf64>) -> tensor<4xf64> {
  // expected-error@below {{diagonals must have a static shape}}
  %0 = kernel.linear_transform %arg0, %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : tensor<4xf64>, tensor<2x?xf64> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// -----

func.func @test_apply_dynamic_input(%arg0: tensor<?xf64>) -> tensor<?xf64> {
  %diagonals = arith.constant dense<1.0> : tensor<2x4xf64>
  %lt = kernel.prepare_linear_transform %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : tensor<2x4xf64> -> !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>
  // expected-error@below {{input must have a static shape}}
  %0 = kernel.apply_linear_transform %arg0, %lt : tensor<?xf64>, !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0> -> tensor<?xf64>
  return %0 : tensor<?xf64>
}

// -----

func.func @test_prepare_dynamic_diagonals(%diagonals: tensor<2x?xf64>) -> !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0> {
  // expected-error@below {{diagonals must have a static shape}}
  %lt = kernel.prepare_linear_transform %diagonals {
    diagonal_indices = array<i64: 0, 1>
  } : tensor<2x?xf64> -> !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>
  return %lt : !kernel.prepared_linear_transform<level = 0, slots = 4, log_bsgs_ratio = 0>
}
