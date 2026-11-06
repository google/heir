// RUN: heir-opt %s --lwe-to-polynomial | FileCheck %s

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
#key = #lwe.key<>
#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>
#modulus_chain_L5_C1_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>
!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>
#ring_rns_L0_1_x8_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**8>>
#ring_rns_L1_1_x8_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**8>>
#ring_Z65537_i64_1_x8_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**8>>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 2>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x8_, encoding = #full_crt_packing_encoding>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x8_, encryption_type = lsb>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x8_, encryption_type = lsb>

!plaintext_rlwe = !lwe.lwe_plaintext<plaintext_space = #plaintext_space>
!ciphertext_rlwe = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>
!ciphertext_rlwe_L1 = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!rlwe_key = !lwe.lwe_secret_key<key = #key, ring = #ring_rns_L0_1_x8_>
!rlwe_key_L1 = !lwe.lwe_secret_key<key = #key, ring = #ring_rns_L1_1_x8_>

#ring_Z1095233372161_1_x8_ = #polynomial.ring<coefficientType = !Z1095233372161_i64_, polynomialModulus = <1 + x**8>>
#ciphertext_space_Z_ = #lwe.ciphertext_space<ring = #ring_Z1095233372161_1_x8_, encryption_type = lsb>
!ciphertext_rlwe_Z = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_Z_, key = #key>
!rlwe_key_Z = !lwe.lwe_secret_key<key = #key, ring = #ring_Z1095233372161_1_x8_>

// CHECK: @test_rlwe_sk_encrypt
// CHECK-SAME: (%[[ARG0:.*]]: ![[POLY_TY:.*]],
// CHECK-SAME:  %[[ARG1:.*]]: tensor<1x![[CIPHERTEXT_POLY_TY:.*]]>)
// CHECK-SAME: -> tensor<2x![[CIPHERTEXT_POLY_TY]]>
func.func @test_rlwe_sk_encrypt(%arg0: !plaintext_rlwe, %arg1: !rlwe_key) -> !ciphertext_rlwe {
  // CHECK-NOT: lwe.rlwe_encrypt

  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CONSTANT_T_ZN:.*]] = mod_arith.constant 65537 : [[MOD_ARITH_TY:.*]]
  // CHECK-DAG: %[[CONSTANT_T:.*]] = mod_arith.mod_switch %[[CONSTANT_T_ZN]] : [[MOD_ARITH_TY]] to [[RNS_TY:.*]]

  // CHECK-DAG: %[[PRNG:.*]] = random.init_prng %[[ZERO]]
  // CHECK-DAG: %[[GAUSSIAN:.*]] = random.discrete_gaussian_distribution %[[PRNG]]

  // Each coefficient of the mask `a` is sampled uniformly modulo each RNS limb.
  // CHECK-DAG: %[[A_COEFFS:.*]] = random.sample_modq %[[PRNG]] : (!random.prng) -> tensor<8x[[RNS_TY]]>
  // CHECK-DAG: %[[A:.*]] = polynomial.from_tensor %[[A_COEFFS]] : tensor<8x[[RNS_TY]]> -> ![[CIPHERTEXT_POLY_TY]]
  // CHECK-DAG: %[[E:.*]] = random.sample %[[GAUSSIAN]] {{.*}} -> ![[CIPHERTEXT_POLY_TY]]

  // CHECK-DAG: %[[SK:.*]] = tensor.extract %[[ARG1]][%[[ZERO]]]

  // c_0 = -a * s + t * e + m
  // CHECK-DAG: %[[A_TIMES_SK:.*]] = polynomial.mul %[[A]], %[[SK]]
  // CHECK-DAG: %[[E_TIMES_T:.*]] = polynomial.mul_scalar %[[E]], %[[CONSTANT_T]]
  // CHECK-DAG: %[[ARG0_CAST:.*]] = polynomial.mod_switch %[[ARG0]]
  // CHECK-DAG: %[[M_PLUS_TE:.*]] = polynomial.add %[[ARG0_CAST]], %[[E_TIMES_T]]
  // CHECK-DAG: %[[C_0:.*]] = polynomial.sub %[[M_PLUS_TE]], %[[A_TIMES_SK]]

  // c_1 = a
  // CHECK:     %[[C:.*]] = tensor.from_elements %[[C_0]], %[[A]]
  // CHECK:     return %[[C]]
  %0 = lwe.rlwe_encrypt %arg0, %arg1 : (!plaintext_rlwe, !rlwe_key) -> !ciphertext_rlwe
  return %0 : !ciphertext_rlwe
}

// With multiple RNS limbs, `a` is still sampled with a single
// `random.sample_modq`, which samples each limb uniformly.

// CHECK: @test_rlwe_sk_encrypt_rns
// CHECK-NOT: lwe.rlwe_encrypt
// CHECK: %[[A_COEFFS:.*]] = random.sample_modq %{{.*}} : (!random.prng) -> tensor<8x!{{.*}}>
// CHECK: polynomial.from_tensor %[[A_COEFFS]]
// CHECK: polynomial.sub
// CHECK: tensor.from_elements
func.func @test_rlwe_sk_encrypt_rns(%arg0: !plaintext_rlwe, %arg1: !rlwe_key_L1) -> !ciphertext_rlwe_L1 {
  %0 = lwe.rlwe_encrypt %arg0, %arg1 : (!plaintext_rlwe, !rlwe_key_L1) -> !ciphertext_rlwe_L1
  return %0 : !ciphertext_rlwe_L1
}

// A plain mod_arith ciphertext coefficient type is treated as a single limb.

// CHECK: @test_rlwe_sk_encrypt_mod_arith
// CHECK-NOT: lwe.rlwe_encrypt
// CHECK: %[[A_COEFFS:.*]] = random.sample_modq %{{.*}} : (!random.prng) -> tensor<8x!{{.*}}>
// CHECK: polynomial.from_tensor %[[A_COEFFS]]
// CHECK: polynomial.sub
// CHECK: tensor.from_elements
func.func @test_rlwe_sk_encrypt_mod_arith(%arg0: !plaintext_rlwe, %arg1: !rlwe_key_Z) -> !ciphertext_rlwe_Z {
  %0 = lwe.rlwe_encrypt %arg0, %arg1 : (!plaintext_rlwe, !rlwe_key_Z) -> !ciphertext_rlwe_Z
  return %0 : !ciphertext_rlwe_Z
}
