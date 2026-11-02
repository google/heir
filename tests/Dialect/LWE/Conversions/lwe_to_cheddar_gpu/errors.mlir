// RUN: not heir-opt --lwe-to-cheddar-gpu --split-input-file %s 2>&1 | FileCheck %s --check-prefix=ERR-BOOT
// RUN: not heir-opt --lwe-to-cheddar-gpu="params=%S/bootparam_40_64bit.yaml" --split-input-file %s 2>&1 | FileCheck %s --check-prefix=ERR-PARAM

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 40>
#key = #lwe.key<>
#ring_f64_1_x65536 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**65536>>

!rns_L2 = !rns.rns<!mod_arith.int<1125899908022273 : i64>, !mod_arith.int<1099515691009 : i64>, !mod_arith.int<1099523555329 : i64>>
#ring_rns_L2 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**65536>>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2, encryption_type = mix>
#chain_L2 = #lwe.modulus_chain<elements = <1125899908022273 : i64, 1099515691009 : i64, 1099523555329 : i64>, current = 2>

!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #chain_L2>

// ERR-BOOT: ckks.bootstrap is not supported
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [1125899908022273, 1099515691009, 1099523555329], P = [72057594038321153], logDefaultScale = 40>} {
  func.func @test_bootstrap(%arg0: !ct_L2) -> !ct_L2 {
    %0 = ckks.bootstrap %arg0 : !ct_L2 -> !ct_L2
    return %0 : !ct_L2
  }
}

// -----

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 40>
#key = #lwe.key<>
#ring_f64_1_x65536 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**65536>>

!rns_L2 = !rns.rns<!mod_arith.int<1125899908022273 : i64>, !mod_arith.int<1099515691009 : i64>, !mod_arith.int<1099523555329 : i64>>
#ring_rns_L2 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**65536>>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2, encryption_type = mix>
#chain_L2 = #lwe.modulus_chain<elements = <1125899908022273 : i64, 1099515691009 : i64, 1099523555329 : i64>, current = 2>

!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #chain_L2>

// ERR-PARAM: parameter mismatch
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 10, Q = [1125899908022273], P = [72057594038321153], logDefaultScale = 30>} {
  func.func @test_mismatch(%arg0: !ct_L2) -> !ct_L2 {
    return %arg0 : !ct_L2
  }
}
