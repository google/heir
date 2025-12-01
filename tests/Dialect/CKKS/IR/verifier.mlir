// RUN: heir-opt --verify-diagnostics --split-input-file %s
!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 1024>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #inverse_canonical_encoding>

#ciphertext_space_L0_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb, size = 3>

!ct = !lwe.lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_D3_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

func.func @test_input_dimension_error(%input: !ct) {
  // expected-error@+1 {{x.dim == 2 does not hold}}
  %out = ckks.rotate  %input { offset = 4 }  : !ct
  return
}

// -----

!Z35184372121601_i64_ = !mod_arith.int<35184372121601 : i64>
!Z36028797019389953_i64_ = !mod_arith.int<36028797019389953 : i64>
// note the scaling factor is 45
// after mul it should be 90
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#modulus_chain_L1_C0_ = #lwe.modulus_chain<elements = <36028797019389953 : i64, 35184372121601 : i64>, current = 0>
#modulus_chain_L1_C1_ = #lwe.modulus_chain<elements = <36028797019389953 : i64, 35184372121601 : i64>, current = 1>
#ring_f64_1_x1024_ = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0_ = !rns.rns<!Z36028797019389953_i64_>
!rns_L1_ = !rns.rns<!Z36028797019389953_i64_, !Z35184372121601_i64_>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb, size = 3>
!ct_L0_ = !lwe.lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = <ring = #ring_f64_1_x1024_, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L1_C0_>
!ct_L1_ = !lwe.lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = <ring = #ring_f64_1_x1024_, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L1_C1_>
!ct_L1_D3_ = !lwe.lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = <ring = #ring_f64_1_x1024_, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1_D3_, key = #key, modulus_chain = #modulus_chain_L1_C1_>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  func.func @mul(%ct: !ct_L1_) -> !ct_L0_ {
    // expected-error@+1 {{'ckks.mul' op output plaintext space does not match}}
    %ct_0 = ckks.mul %ct, %ct : (!ct_L1_, !ct_L1_) -> !ct_L1_D3_
    %ct_1 = ckks.relinearize %ct_0 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L1_D3_) -> !ct_L1_
    %ct_2 = ckks.rescale %ct_1 {to_ring = #ring_rns_L0_1_x1024_} : !ct_L1_ -> !ct_L0_
    return %ct_2 : !ct_L0_
  }
}

// -----

!Z35184372121601_i64_ = !mod_arith.int<35184372121601 : i64>
!Z36028797019389953_i64_ = !mod_arith.int<36028797019389953 : i64>
// note the scaling factor is 45
// after mul it should be 90
// after rescaling it should be 45
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#inverse_canonical_encoding1 = #lwe.inverse_canonical_encoding<scaling_factor = 90>
#inverse_canonical_encoding2 = #lwe.inverse_canonical_encoding<scaling_factor = 40>
#key = #lwe.key<>
#modulus_chain_L1_C0_ = #lwe.modulus_chain<elements = <36028797019389953 : i64, 35184372121601 : i64>, current = 0>
#modulus_chain_L1_C1_ = #lwe.modulus_chain<elements = <36028797019389953 : i64, 35184372121601 : i64>, current = 1>
#ring_f64_1_x1024_ = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0_ = !rns.rns<!Z36028797019389953_i64_>
!rns_L1_ = !rns.rns<!Z36028797019389953_i64_, !Z35184372121601_i64_>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb, size = 3>
!ct_L0_ = !lwe.lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = <ring = #ring_f64_1_x1024_, encoding = #inverse_canonical_encoding2>, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L1_C0_>
!ct_L1_ = !lwe.lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = <ring = #ring_f64_1_x1024_, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L1_C1_>
!ct_L1_1 = !lwe.lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = <ring = #ring_f64_1_x1024_, encoding = #inverse_canonical_encoding1>, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L1_C1_>
!ct_L1_D3_ = !lwe.lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = <ring = #ring_f64_1_x1024_, encoding = #inverse_canonical_encoding1>, ciphertext_space = #ciphertext_space_L1_D3_, key = #key, modulus_chain = #modulus_chain_L1_C1_>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  func.func @mul(%ct: !ct_L1_) -> !ct_L0_ {
    %ct_0 = ckks.mul %ct, %ct : (!ct_L1_, !ct_L1_) -> !ct_L1_D3_
    %ct_1 = ckks.relinearize %ct_0 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_L1_D3_) -> !ct_L1_1
    // expected-error@+1 {{'ckks.rescale' op output plaintext space does not match}}
    %ct_2 = ckks.rescale %ct_1 {to_ring = #ring_rns_L0_1_x1024_} : !ct_L1_1 -> !ct_L0_
    return %ct_2 : !ct_L0_
  }
}

// -----

!Z35184372121601_i64_ = !mod_arith.int<35184372121601 : i64>
!Z36028797019389953_i64_ = !mod_arith.int<36028797019389953 : i64>
// note the scaling factor is 45
// after mul it should be 90
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#modulus_chain_L1_C0_ = #lwe.modulus_chain<elements = <36028797019389953 : i64, 35184372121601 : i64>, current = 0>
#modulus_chain_L1_C1_ = #lwe.modulus_chain<elements = <36028797019389953 : i64, 35184372121601 : i64>, current = 1>
#ring_f64_1_x1024_ = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0_ = !rns.rns<!Z36028797019389953_i64_>
!rns_L1_ = !rns.rns<!Z36028797019389953_i64_, !Z35184372121601_i64_>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb>
!ct_L0_ = !lwe.lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = <ring = #ring_f64_1_x1024_, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L1_C0_>
!ct_L1_ = !lwe.lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = <ring = #ring_f64_1_x1024_, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L1_C1_>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  func.func @bootstrap(%ct: !ct_L0_) -> !ct_L1_ {
    // expected-error@+1 {{'ckks.bootstrap' op output ciphertext must have 2 levels but has 1}}
    %ct_2 = ckks.bootstrap %ct {targetLevel = 2} : !ct_L0_ -> !ct_L1_
    return %ct_2 : !ct_L1_
  }
}
