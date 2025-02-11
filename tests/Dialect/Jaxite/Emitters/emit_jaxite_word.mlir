// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>

#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 1024>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #full_crt_packing_encoding>
#plaintext_space_f16 = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #inverse_canonical_encoding>

!pt = !lwe.new_lwe_plaintext<application_data = <message_type = i3>, plaintext_space = #plaintext_space>
!ptf16 = !lwe.new_lwe_plaintext<application_data = <message_type = f16>, plaintext_space = #plaintext_space_f16>

#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb>

!pk = !openfhe.public_key
!sk = !openfhe.private_key
!ek = !openfhe.eval_key
!cc = !openfhe.crypto_context
!ct = !lwe.new_lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

module attributes {scheme.ckks} {
  // find functions here: third_party/heir/tests/Dialect/Openfhe/IR/ops.mlir
  func.func @test_add(%cc : !cc, %ct : !ct) -> !ct {
    %out = openfhe.add %cc, %ct, %ct: (!cc, !ct, !ct) -> !ct
    return %out : !ct
  }
}
