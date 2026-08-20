// RUN: heir-opt --split-input-file --prepare-linear-transforms %s | FileCheck %s

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0 = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = mix>
!ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain>

// The ciphertext's chain has current 0, so it sits at level 0; its ring has
// degree 1024 with inverse-canonical encoding, so 512 slots.

// CHECK: @split
// CHECK: %[[LT:.*]] = kernel.prepare_linear_transform %{{.*}} {diagonal_indices = array<i64: 0, 2>, source_row_indices = array<i64: 1, 3>} : tensor<4x512xf64> -> <level = 0, slots = 512, log_bsgs_ratio = 0>
// CHECK: %[[OUT:.*]] = kernel.apply_linear_transform %{{.*}}, %[[LT]] {kernel.test} : {{.*}}<level = 0, slots = 512, log_bsgs_ratio = 0>{{.*}}
// CHECK-NOT: kernel.linear_transform
// CHECK: return %[[OUT]]
module attributes {backend.lattigo, scheme.ckks} {
  func.func @split(%ct: !ct) -> !ct {
    %diagonals = arith.constant dense<1.0> : tensor<4x512xf64>
    %0 = kernel.linear_transform %ct, %diagonals {diagonal_indices = array<i64: 0, 2>, kernel.test, source_row_indices = array<i64: 1, 3>} : !ct, tensor<4x512xf64> -> !ct
    return %0 : !ct
  }
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

// A target that does not declare has_prepared_linear_transform keeps the
// sugar form.

// CHECK: @keeps_sugar
// CHECK: kernel.linear_transform
// CHECK-NOT: kernel.prepare_linear_transform
module attributes {backend.openfhe, scheme.ckks} {
  func.func @keeps_sugar(%ct: !ct) -> !ct {
    %diagonals = arith.constant dense<1.0> : tensor<2x512xf64>
    %0 = kernel.linear_transform %ct, %diagonals {diagonal_indices = array<i64: 0, 1>} : !ct, tensor<2x512xf64> -> !ct
    return %0 : !ct
  }
}
