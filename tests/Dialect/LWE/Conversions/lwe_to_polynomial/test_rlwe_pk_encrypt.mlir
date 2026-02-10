// RUN: heir-opt %s --lwe-to-polynomial | FileCheck %s

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
#key = #lwe.key<>
#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>
!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
#ring_rns_L0_1_x8_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**8>>
#ring_Z65537_i64_1_x8_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**8>>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 2>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x8_, encoding = #full_crt_packing_encoding>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x8_, encryption_type = lsb>

!plaintext_rlwe = !lwe.lwe_plaintext<plaintext_space = #plaintext_space>
!ciphertext_rlwe = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>
!rlwe_key = !lwe.lwe_public_key<key = #key, ring = #ring_rns_L0_1_x8_>

// CHECK: @test_rlwe_pk_encrypt
// CHECK-SAME: (%[[ARG0:.*]]: ![[POLY_TY:.*]],
// CHECK-SAME:  %[[ARG1:.*]]: tensor<2x![[CIPHERTEXT_POLY_TY:.*]]>)
// CHECK-SAME: -> tensor<2x![[CIPHERTEXT_POLY_TY:.*]]>
func.func @test_rlwe_pk_encrypt(%arg0: !plaintext_rlwe, %arg1: !rlwe_key) -> !ciphertext_rlwe {
  // CHECK-NOT: lwe.rlwe_encrypt

  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CONSTANT_T_ZN:.*]] = mod_arith.constant 65537 : [[MOD_ARITH_TY:.*]]
  // CHECK-DAG: %[[CONSTANT_T:.*]] = mod_arith.mod_switch %[[CONSTANT_T_ZN]] : [[MOD_ARITH_TY]] to [[RNS_TY:.*]]

  // CHECK-DAG: %[[PRNG:.*]] = random.init_prng %[[SEED:.*]]
  // CHECK-DAG: %[[UNIFORM:.*]] = random.discrete_uniform_distribution %[[PRNG]]
  // CHECK-DAG: %[[GAUSSIAN:.*]] = random.discrete_gaussian_distribution %[[PRNG]]

  // CHECK-DAG: %[[PK_0:.*]] = tensor.extract %[[ARG1]][%[[ZERO]]]
  // CHECK-DAG: %[[PK_1:.*]] = tensor.extract %[[ARG1]][%[[ONE]]]

  // CHECK-DAG: %[[U:.*]] = random.sample %[[UNIFORM]]
  // CHECK:     %[[E_0:.*]] = random.sample %[[GAUSSIAN]]
  // CHECK:     %[[E_1:.*]] = random.sample %[[GAUSSIAN]]

  // CHECK-DAG: %[[ARG0_CAST:.*]] = polynomial.mod_switch %[[ARG0]]
  // CHECK-DAG: %[[PK_0_TIMES_U:.*]] = polynomial.mul %[[PK_0]], %[[U]]
  // CHECK-DAG: %[[E_0_TIMES_T:.*]] = polynomial.mul_scalar %[[E_0]], %[[CONSTANT_T]]
  // CHECK-DAG: %[[PK_0_TIMES_U_PLUS_E_0_TIMES_T:.*]] = polynomial.add %[[PK_0_TIMES_U]], %[[E_0_TIMES_T]]
  // CHECK-DAG: %[[C_0:.*]] = polynomial.add %[[PK_0_TIMES_U_PLUS_E_0_TIMES_T]], %[[ARG0_CAST]]

  // CHECK-DAG: %[[PK_1_TIMES_U:.*]] = polynomial.mul %[[PK_1]], %[[U]]
  // CHECK-DAG: %[[E_1_TIMES_T:.*]] = polynomial.mul_scalar %[[E_1]], %[[CONSTANT_T]]
  // CHECK-DAG: %[[C_1:.*]] = polynomial.add %[[PK_1_TIMES_U]], %[[E_1_TIMES_T]]

  // CHECK:     %[[C:.*]] = tensor.from_elements %[[C_0]], %[[C_1]]
  // CHECK:     return %[[C]]
  %0 = lwe.rlwe_encrypt %arg0, %arg1 : (!plaintext_rlwe, !rlwe_key) -> !ciphertext_rlwe
  return %0 : !ciphertext_rlwe
}
