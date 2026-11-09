// RUN: heir-opt --verify-diagnostics --split-input-file %s 2>&1

#poly = #polynomial.int_polynomial<x>
#key = #lwe.key<slot_index = 0>
#pspace = #lwe.plaintext_space<
  ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>,
  encoding = #lwe.constant_coefficient_encoding<scaling_factor = 256>>
!plaintext = !lwe.lwe_plaintext<plaintext_space = #pspace>
func.func @test_invalid_plaintext_bits() {
    %0 = arith.constant 0 : i1
    // expected-error@below {{LWE plaintext ring coefficient type width must match message bits parameter, expected 4 but got 3}}
    %2 = lwe.encode %0 <plaintext_bits = 4>: i1 to !plaintext
  return
}

// -----

#poly = #polynomial.int_polynomial<x>
#pspace = #lwe.plaintext_space<
  ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>,
  encoding = #lwe.constant_coefficient_encoding<scaling_factor = 256>>
!plaintext = !lwe.lwe_plaintext<plaintext_space = #pspace>

#key = #lwe.key<slot_index = 0>
#diff_pspace = #lwe.plaintext_space<
  ring = #polynomial.ring<coefficientType = i3, polynomialModulus = #poly>,
  encoding = #lwe.constant_coefficient_encoding<scaling_factor = 512>>

#cspace = #lwe.ciphertext_space<
  ring = #polynomial.ring<coefficientType = i32, polynomialModulus = #poly>,
  encryption_type = msb, size = 742>
!ciphertext = !lwe.lwe_ciphertext<plaintext_space = #diff_pspace, ciphertext_space = #cspace, key = #key>

func.func @test_invalid_mismatch_encoding(%1: !plaintext) {
    // expected-error@below {{op failed to verify that the first arg's type's encoding matches the given encoding}}
    %2 = lwe.trivial_encrypt %1 : !plaintext -> !ciphertext
  return
}

// -----

!Zq0 = !mod_arith.int<1095233372161 : i64>
!Zq1 = !mod_arith.int<1032955396097 : i64>
!Zp0 = !mod_arith.int<261405424692085787 : i64>

#key = #lwe.key<>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#ring_L1 = #polynomial.ring<coefficientType = !rns.rns<!Zq0>, polynomialModulus = <1 + x**1024>>
#ring_L1_other = #polynomial.ring<coefficientType = !rns.rns<!Zq1>, polynomialModulus = <1 + x**1024>>
#ring_L2 = #polynomial.ring<coefficientType = !rns.rns<!Zq0, !Zp0>, polynomialModulus = <1 + x**1024>>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_L1, encoding = #inverse_canonical_encoding>, ciphertext_space = <ring = #ring_L1_other, encryption_type = lsb>, key = #key>
!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_L1, encoding = #inverse_canonical_encoding>, ciphertext_space = <ring = #ring_L2, encryption_type = lsb>, key = #key>

func.func @test_mod_down_not_prefix(%x: !ct_L2) -> !ct_L1 {
  // expected-error@below {{Target basis is not a prefix of the input basis}}
  %0 = lwe.mod_down %x <targetBasis = !rns.rns<!Zq1>> : !ct_L2 -> !ct_L1
  return %0 : !ct_L1
}

// -----

!Zq0 = !mod_arith.int<1095233372161 : i64>
!Zp0 = !mod_arith.int<261405424692085787 : i64>

#key = #lwe.key<>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#ring_L1 = #polynomial.ring<coefficientType = !rns.rns<!Zq0>, polynomialModulus = <1 + x**1024>>
#ring_L2 = #polynomial.ring<coefficientType = !rns.rns<!Zq0, !Zp0>, polynomialModulus = <1 + x**1024>>
!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_L1, encoding = #inverse_canonical_encoding>, ciphertext_space = <ring = #ring_L2, encryption_type = lsb>, key = #key>

func.func @test_mod_down_same_basis(%x: !ct_L2) -> !ct_L2 {
  // expected-error@below {{Target basis is not a prefix of the input basis}}
  %0 = lwe.mod_down %x <targetBasis = !rns.rns<!Zq0, !Zp0>> : !ct_L2 -> !ct_L2
  return %0 : !ct_L2
}
