!Z1005037682689_i64_ = !mod_arith.int<1005037682689 : i64>
!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 1024>
#key = #lwe.key<>
#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>
#modulus_chain_L5_C1_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>
#modulus_chain_L5_C2_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 2>
!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>
!rns_L2_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_, !Z1005037682689_i64_>
#ring_Z65537_i64_1_x32_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**32>>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32_, encoding = #inverse_canonical_encoding>
#ring_rns_L0_1_x32_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**32>>
#ring_rns_L1_1_x32_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**32>>
#ring_rns_L2_1_x32_ = #polynomial.ring<coefficientType = !rns_L2_, polynomialModulus = <1 + x**32>>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x32_, encryption_type = lsb>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x32_, encryption_type = lsb>
#ciphertext_space_L2_ = #lwe.ciphertext_space<ring = #ring_rns_L2_1_x32_, encryption_type = lsb>

!ct_L0_ = !lwe.lwe_ciphertext<application_data = <message_type = tensor<8xf64>>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>
!ct_L1_ = !lwe.lwe_ciphertext<application_data = <message_type = tensor<8xf64>>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct_L2_ = !lwe.lwe_ciphertext<application_data = <message_type = tensor<8xf64>>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L2_, key = #key, modulus_chain = #modulus_chain_L5_C2_>

module attributes {scheme.ckks} {
  func.func @simple_ckks_bootstrapping(%cc: !openfhe.crypto_context, %ct: !ct_L2_) -> !ct_L2_ {
    // in FLEXIBLEAUTOEXT mode, openfhe won't execute those mod_reduce
    // added here just for type conversion
    %0 = openfhe.mod_reduce %cc, %ct : (!openfhe.crypto_context, !ct_L2_) -> !ct_L1_
    %1 = openfhe.mod_reduce %cc, %0 : (!openfhe.crypto_context, !ct_L1_) -> !ct_L0_
    %2 = openfhe.bootstrap %cc, %1 : (!openfhe.crypto_context, !ct_L0_) -> !ct_L2_
    return %2 : !ct_L2_
  }
}
