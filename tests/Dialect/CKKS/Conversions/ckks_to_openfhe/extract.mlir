// RUN: heir-opt --ckks-to-lwe --lwe-to-openfhe %s | FileCheck %s

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>

#ring_Z65537_i64_1_x32_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**32>>
#ring_rns_L0_1_x32_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**32>>

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32_, encoding = #inverse_canonical_encoding>

#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x32_, encryption_type = lsb>

!ty1 = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<32xi16>>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>
!ty2 = !lwe.new_lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

// CHECK-LABEL: @test_lower_extract
// CHECK-SAME: %[[arg0:.*]]:
func.func @test_lower_extract(%arg0: !ty1) -> !ty2 {
  // CHECK: arith.constant dense<[0, 0, 0, 0, 1, [[unused:[0 ,]*]]]> : tensor<32xi16>
  // CHECK: openfhe.make_ckks_packed_plaintext
  // CHECK: openfhe.mul_plain
  // CHECK: openfhe.rot
  // CHECK: lwe.reinterpret_application_data
  // CHECK: return
  %c4 = arith.constant 4 : index
  %0 = ckks.extract %arg0, %c4 : (!ty1, index) -> !ty2
  return %0 : !ty2
}
