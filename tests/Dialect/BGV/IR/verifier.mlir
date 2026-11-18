// RUN: heir-opt --verify-diagnostics --split-input-file %s

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>


#ciphertext_space_L0_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb, size = 3>

!ct = !lwe.lwe_ciphertext<encoding = #full_crt_packing_encoding, ciphertext_space = #ciphertext_space_L0_D3_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

func.func @test_input_dimension_error(%input: !ct) {
  // expected-error@+1 {{x.dim == 2 does not hold}}
  %out = bgv.rotate_cols  %input <static_shift = 4> : !ct
  return
}

// -----

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #full_crt_packing_encoding>

#ciphertext_space_L0_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb, size = 3>

!pt = !lwe.lwe_plaintext<plaintext_space = #plaintext_space>
!ct = !lwe.lwe_ciphertext<encoding = #full_crt_packing_encoding, ciphertext_space = #ciphertext_space_L0_D3_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

func.func @add_plain_ct_ct(%input: !ct) {
  // expected-error@+1 {{expected ciphertext, plaintext operand types}}
  %out = bgv.add_plain  %input, %input : (!ct, !ct) -> !ct
  return
}

// -----

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #full_crt_packing_encoding>
#ciphertext_space_L0_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb, size = 3>

!pt = !lwe.lwe_plaintext<plaintext_space = #plaintext_space>
!ct = !lwe.lwe_ciphertext<encoding = #full_crt_packing_encoding, ciphertext_space = #ciphertext_space_L0_D3_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

func.func @add_plain_pt_pt(%input: !pt) {
  // expected-error@+1 {{expected plaintext, ciphertext operand types}}
  %out = bgv.add_plain  %input, %input : (!pt, !pt) -> !ct
  return
}

// -----

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #full_crt_packing_encoding>
#ciphertext_space_L0_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb, size = 3>
#ciphertext_space_L0_D3_msb_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = msb, size = 3>

!pt = !lwe.lwe_plaintext<plaintext_space = #plaintext_space>
!ct = !lwe.lwe_ciphertext<encoding = #full_crt_packing_encoding, ciphertext_space = #ciphertext_space_L0_D3_, key = #key, modulus_chain = #modulus_chain_L5_C0_>
!ct1 = !lwe.lwe_ciphertext<encoding = #full_crt_packing_encoding, ciphertext_space = #ciphertext_space_L0_D3_msb_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

func.func @add_plain_result(%input: !pt, %input1: !ct) {
  // expected-error@+1 {{op requires all ciphertexts to have the same ciphertextType}}
  %out = bgv.add_plain  %input, %input1 : (!pt, !ct) -> !ct1
  return
}

// -----

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#coeff_encoding = #lwe.constant_coefficient_encoding<scaling_factor=10000>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #full_crt_packing_encoding>
#plaintext_space1 = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #coeff_encoding>
#ciphertext_space_L0_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb, size = 3>

!pt = !lwe.lwe_plaintext<plaintext_space = #plaintext_space>
!ct = !lwe.lwe_ciphertext<encoding = #full_crt_packing_encoding, ciphertext_space = #ciphertext_space_L0_D3_, key = #key, modulus_chain = #modulus_chain_L5_C0_>
!pt1 = !lwe.lwe_plaintext<plaintext_space = #plaintext_space1>

func.func @add_plain_result(%input: !pt1, %input1: !ct) {
  // expected-error@+1 {{requires all operands and results to have the same encoding}}
  %out = bgv.add_plain  %input, %input1 : (!pt1, !ct) -> !ct
  return
}

// -----

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
#ring_Z65537_i64_1_x512_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**512>>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#plaintext_space_512 = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x512_, encoding = #full_crt_packing_encoding>
#ciphertext_space_L0_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb, size = 3>

!pt_512 = !lwe.lwe_plaintext<plaintext_space = #plaintext_space_512>
!ct = !lwe.lwe_ciphertext<encoding = #full_crt_packing_encoding, ciphertext_space = #ciphertext_space_L0_D3_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

func.func @add_plain_mismatched_ring_degree(%input: !pt_512, %input1: !ct) {
  // expected-error@+1 {{requires plaintext ring polynomial modulus to match the ciphertext ring polynomial modulus}}
  %out = bgv.add_plain %input, %input1 : (!pt_512, !ct) -> !ct
  return
}
