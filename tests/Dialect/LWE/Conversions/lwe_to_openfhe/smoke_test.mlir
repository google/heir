// RUN: heir-opt --lwe-to-openfhe %s

!Z34359754753_i64 = !mod_arith.int<34359754753 : i64>
!Z65537_i64 = !mod_arith.int<65537 : i64>
!Z67239937_i64 = !mod_arith.int<67239937 : i64>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 1>
#key = #lwe.key<>
#modulus_chain_L1_C1 = #lwe.modulus_chain<elements = <67239937 : i64, 34359754753 : i64>, current = 1>
!rns_L1 = !rns.rns<!Z67239937_i64, !Z34359754753_i64>
#ring_Z65537_i64_1_x32 = #polynomial.ring<coefficientType = !Z65537_i64, polynomialModulus = <1 + x**32>>
#ring_rns_L1_1_x32 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**32>>
!pkey_L1 = !lwe.new_lwe_public_key<key = #key, ring = #ring_rns_L1_1_x32>
!pt = !lwe.new_lwe_plaintext<application_data = <message_type = tensor<32xi16>>, plaintext_space = <ring = #ring_Z65537_i64_1_x32, encoding = #full_crt_packing_encoding>>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x32, encryption_type = lsb>
!ct_L1 = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<32xi16>>, plaintext_space = <ring = #ring_Z65537_i64_1_x32, encoding = #full_crt_packing_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>
module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 12, Q = [67239937, 34359754753], P = [34359771137], plaintextModulus = 65537>, scheme.bgv} {
  func.func @simple_sum__encrypt__arg0(%arg0: tensor<32xi16>, %pk: !pkey_L1) -> !ct_L1 {
    %cst = arith.constant dense<0> : tensor<4096xi16>
    %pt = lwe.rlwe_encode %cst {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32} : tensor<4096xi16> -> !pt
    %ct = lwe.rlwe_encrypt %pt, %pk : (!pt, !pkey_L1) -> !ct_L1
    return %ct : !ct_L1
  }
}
