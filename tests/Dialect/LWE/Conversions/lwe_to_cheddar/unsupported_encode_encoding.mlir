// RUN: not heir-opt --lwe-to-cheddar %s 2>&1 | FileCheck %s

// CHECK: error: 'lwe.rlwe_encode' op requires inverse-canonical CKKS plaintext encoding for CHEDDAR lowering

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
!Z65537_i64 = !mod_arith.int<65537 : i64>
!rns_L0 = !rns.rns<!Z65537_i64>
#ring_Z65537_i64_1_x32 = #polynomial.ring<coefficientType = !Z65537_i64, polynomialModulus = <1 + x**32>>
#ring_rns_L0_1_x32 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**32>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x32, encryption_type = mix>
#modulus_chain_L0_C0 = #lwe.modulus_chain<elements = <65537 : i64>, current = 0>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32, encoding = #full_crt_packing_encoding>
!pt = !lwe.lwe_plaintext<plaintext_space = #plaintext_space>
!ct_L0 = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain_L0_C0>

module attributes {scheme.ckks, ckks.schemeParam = #ckks.scheme_param<logN = 5, Q = [65537], P = [114689], logDefaultScale = 4>} {
  func.func @bad_encode(%ct: !ct_L0) -> !ct_L0 {
    %cst = arith.constant dense<1> : tensor<32xi16>
    %pt = lwe.rlwe_encode %cst {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32} : tensor<32xi16> -> !pt
    %result = lwe.radd_plain %ct, %pt : (!ct_L0, !pt) -> !ct_L0
    return %result : !ct_L0
  }
}
