// RUN: heir-opt %s --lwe-to-polynomial | FileCheck %s
// RUN: heir-opt %s --lwe-to-polynomial | FileCheck %s --check-prefix=CKKS

// CKKS encryption and decryption. The plaintext is an RNS polynomial holding
// the encoded (already scaled) message, and unlike BGV, the encryption noise
// is not scaled by a plaintext modulus, and the plaintext is not converted to
// the ciphertext ring. The CKKS-NOT checks below (run separately, so that they
// apply to the entire output) enforce this for every function in this file.

// CKKS-NOT: mul_scalar
// CKKS-NOT: mod_switch
// CKKS-NOT: mod_arith.constant

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
#key = #lwe.key<>
!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
#ring_rns_L0_1_x8_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**8>>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 40>
#plaintext_space_L0_ = #lwe.plaintext_space<ring = #ring_rns_L0_1_x8_, encoding = #inverse_canonical_encoding>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x8_, encryption_type = ckks>

!pt_L0 = !lwe.lwe_plaintext<plaintext_space = #plaintext_space_L0_>
!ct = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space_L0_, ciphertext_space = #ciphertext_space_L0_, key = #key>
!sk = !lwe.lwe_secret_key<key = #key, ring = #ring_rns_L0_1_x8_>
!pk = !lwe.lwe_public_key<key = #key, ring = #ring_rns_L0_1_x8_>

// CHECK: @test_ckks_sk_encrypt
// CHECK-SAME: (%[[ARG0:.*]]: ![[POLY_TY:.*]],
// CHECK-SAME:  %[[ARG1:.*]]: tensor<1x![[POLY_TY]]>)
// CHECK-SAME: -> tensor<2x![[POLY_TY]]>
func.func @test_ckks_sk_encrypt(%arg0: !pt_L0, %arg1: !sk) -> !ct {
  // CHECK-NOT: lwe.rlwe_encrypt

  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[PRNG:.*]] = random.init_prng %[[ZERO]]
  // CHECK-DAG: %[[GAUSSIAN:.*]] = random.discrete_gaussian_distribution %[[PRNG]]
  // CHECK-DAG: %[[A_COEFFS:.*]] = random.sample_modq %[[PRNG]]
  // CHECK-DAG: %[[A:.*]] = polynomial.from_tensor %[[A_COEFFS]]
  // CHECK-DAG: %[[E:.*]] = random.sample %[[GAUSSIAN]]
  // CHECK-DAG: %[[SK:.*]] = tensor.extract %[[ARG1]][%[[ZERO]]]

  // c_0 = -a * s + e + m
  // CHECK-DAG: %[[A_TIMES_SK:.*]] = polynomial.mul %[[A]], %[[SK]]
  // CHECK-DAG: %[[M_PLUS_E:.*]] = polynomial.add %[[ARG0]], %[[E]]
  // CHECK-DAG: %[[C_0:.*]] = polynomial.sub %[[M_PLUS_E]], %[[A_TIMES_SK]]

  // c_1 = a
  // CHECK:     %[[C:.*]] = tensor.from_elements %[[C_0]], %[[A]]
  // CHECK:     return %[[C]]
  %0 = lwe.rlwe_encrypt %arg0, %arg1 : (!pt_L0, !sk) -> !ct
  return %0 : !ct
}

// CHECK: @test_ckks_pk_encrypt
// CHECK-SAME: (%[[ARG0:.*]]: ![[POLY_TY:.*]],
// CHECK-SAME:  %[[ARG1:.*]]: tensor<2x![[POLY_TY]]>)
// CHECK-SAME: -> tensor<2x![[POLY_TY]]>
func.func @test_ckks_pk_encrypt(%arg0: !pt_L0, %arg1: !pk) -> !ct {
  // CHECK-NOT: lwe.rlwe_encrypt

  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[PRNG:.*]] = random.init_prng %[[ZERO]]
  // CHECK-DAG: %[[UNIFORM:.*]] = random.discrete_uniform_distribution %[[PRNG]]
  // CHECK-DAG: %[[GAUSSIAN:.*]] = random.discrete_gaussian_distribution %[[PRNG]]
  // CHECK-DAG: %[[U:.*]] = random.sample %[[UNIFORM]]
  // CHECK-DAG: %[[PK_0:.*]] = tensor.extract %[[ARG1]][%[[ZERO]]]
  // CHECK-DAG: %[[PK_1:.*]] = tensor.extract %[[ARG1]][%[[ONE]]]
  // CHECK:     %[[E_0:.*]] = random.sample %[[GAUSSIAN]]
  // CHECK:     %[[E_1:.*]] = random.sample %[[GAUSSIAN]]

  // c_0 = pk_0 * u + e_0 + m
  // CHECK-DAG: %[[PK_0_TIMES_U:.*]] = polynomial.mul %[[PK_0]], %[[U]]
  // CHECK-DAG: %[[PK_0_TIMES_U_PLUS_E_0:.*]] = polynomial.add %[[PK_0_TIMES_U]], %[[E_0]]
  // CHECK-DAG: %[[C_0:.*]] = polynomial.add %[[PK_0_TIMES_U_PLUS_E_0]], %[[ARG0]]

  // c_1 = pk_1 * u + e_1
  // CHECK-DAG: %[[PK_1_TIMES_U:.*]] = polynomial.mul %[[PK_1]], %[[U]]
  // CHECK-DAG: %[[C_1:.*]] = polynomial.add %[[PK_1_TIMES_U]], %[[E_1]]

  // CHECK:     %[[C:.*]] = tensor.from_elements %[[C_0]], %[[C_1]]
  // CHECK:     return %[[C]]
  %0 = lwe.rlwe_encrypt %arg0, %arg1 : (!pt_L0, !pk) -> !ct
  return %0 : !ct
}

// CHECK: @test_ckks_decrypt
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x![[POLY_TY:.*]]>,
// CHECK-SAME:  %[[ARG1:.*]]: tensor<1x![[POLY_TY]]>)
// CHECK-SAME: -> ![[POLY_TY]]
func.func @test_ckks_decrypt(%arg0: !ct, %arg1: !sk) -> !pt_L0 {
  // CHECK-NOT: lwe.rlwe_decrypt

  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C_0:.*]] = tensor.extract %[[ARG0]][%[[ZERO]]]
  // CHECK-DAG: %[[C_1:.*]] = tensor.extract %[[ARG0]][%[[ONE]]]
  // CHECK-DAG: %[[SK:.*]] = tensor.extract %[[ARG1]][%[[ZERO]]]

  // m + e = c_0 + c_1 * s
  // CHECK:     %[[C_1_TIMES_SK:.*]] = polynomial.mul %[[C_1]], %[[SK]]
  // CHECK:     %[[PT:.*]] = polynomial.add %[[C_0]], %[[C_1_TIMES_SK]]
  // CHECK:     return %[[PT]]
  %0 = lwe.rlwe_decrypt %arg0, %arg1 : (!ct, !sk) -> !pt_L0
  return %0 : !pt_L0
}

// CKKS ciphertexts with a single (non-RNS) coefficient modulus are also
// supported.
!Z1095233372161_i64_single = !mod_arith.int<1095233372161 : i64>
#ring_Z_1_x8_ = #polynomial.ring<coefficientType = !Z1095233372161_i64_single, polynomialModulus = <1 + x**8>>
#plaintext_space_Z_ = #lwe.plaintext_space<ring = #ring_Z_1_x8_, encoding = #inverse_canonical_encoding>
#ciphertext_space_Z_ = #lwe.ciphertext_space<ring = #ring_Z_1_x8_, encryption_type = ckks>
!pt_Z = !lwe.lwe_plaintext<plaintext_space = #plaintext_space_Z_>
!ct_Z = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space_Z_, ciphertext_space = #ciphertext_space_Z_, key = #key>
!sk_Z = !lwe.lwe_secret_key<key = #key, ring = #ring_Z_1_x8_>

// CHECK: @test_ckks_decrypt_single_modulus
// CHECK:     %[[C_1_TIMES_SK:.*]] = polynomial.mul
// CHECK:     %[[PT:.*]] = polynomial.add %{{.*}}, %[[C_1_TIMES_SK]]
// CHECK:     return %[[PT]]
func.func @test_ckks_decrypt_single_modulus(%arg0: !ct_Z, %arg1: !sk_Z) -> !pt_Z {
  %0 = lwe.rlwe_decrypt %arg0, %arg1 : (!ct_Z, !sk_Z) -> !pt_Z
  return %0 : !pt_Z
}
