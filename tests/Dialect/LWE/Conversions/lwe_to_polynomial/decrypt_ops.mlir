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
!plaintext_rlwe = !lwe.lwe_plaintext<application_data = <message_type = f16>, plaintext_space = #plaintext_space>
!ciphertext_rlwe = !lwe.lwe_ciphertext<application_data = <message_type = f16>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

!rlwe_key = !lwe.lwe_secret_key<key = #key, ring = #ring_rns_L0_1_x8_>

func.func @test_rlwe_decrypt(%arg0: !ciphertext_rlwe, %arg1: !rlwe_key) -> !plaintext_rlwe {
  // CHECK-NOT: lwe.rlwe_decrypt

  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CTXT_ZERO:.*]] = tensor.extract %arg0[%[[ZERO]]]
  // CHECK-DAG: %[[CTXT_ONE:.*]] = tensor.extract %arg0[%[[ONE]]]
  // CHECK-DAG: %[[SECRET_KEY:.*]] = tensor.extract %arg1[%[[ZERO]]]

  // CHECK:     %[[KEY_TIMES_CTXT_ZERO:.*]] = polynomial.mul %[[SECRET_KEY]], %[[CTXT_ZERO]]
  // CHECK:     %[[DECRYPTED_PTXT:.*]] = polynomial.add %[[KEY_TIMES_CTXT_ZERO]], %[[CTXT_ONE]]
  // CHECK:     %[[DECRYPTED_PTXT_MOD_SWITCH:.*]] = polynomial.mod_switch %[[DECRYPTED_PTXT]]
  // CHECK:     return %[[DECRYPTED_PTXT_MOD_SWITCH]]
  %0 = lwe.rlwe_decrypt %arg0, %arg1 : (!ciphertext_rlwe, !rlwe_key) -> !plaintext_rlwe
  return %0 : !plaintext_rlwe
}
