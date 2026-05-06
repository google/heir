// RUN: heir-opt %s | FileCheck %s
// These tests are just checking syntax for RLWE ops.

#encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#poly_1024 = #polynomial.int_polynomial<1 + x**1024>
!plaintext_coefficient_modulus = !mod_arith.int<65537 : i64>
#plaintext_ring = #polynomial.ring<
  coefficientType=!plaintext_coefficient_modulus,
  polynomialModulus=#poly_1024>
#plaintext_space = #lwe.plaintext_space<
  ring=#plaintext_ring,
  encoding=#encoding>
!ciphertext_coefficient_modulus = !mod_arith.int<1032955396097 : i64>
#ciphertext_ring = #polynomial.ring<
  coefficientType=!ciphertext_coefficient_modulus,
  polynomialModulus=#poly_1024>
#ciphertext_space = #lwe.ciphertext_space<
  ring=#ciphertext_ring,
  encryption_type=lsb,
  size=2>
#post_mul_ciphertext_space = #lwe.ciphertext_space<
  ring=#ciphertext_ring,
  encryption_type=lsb,
  size=3>
!ct = !lwe.lwe_ciphertext<
  plaintext_space=#plaintext_space,
  ciphertext_space=#ciphertext_space,
  key=#lwe.key<>>
!post_mul_ct = !lwe.lwe_ciphertext<
  plaintext_space=#plaintext_space,
  ciphertext_space=#post_mul_ciphertext_space,
  key=#lwe.key<>>

!pt = !lwe.lwe_plaintext<
  plaintext_space=#plaintext_space>

// CHECK: test_radd
func.func @test_radd(%0: !ct, %1: !ct) -> !ct {
  // CHECK: lwe.radd
  %2 = lwe.radd %0, %1 : (!ct, !ct) -> !ct
  return %2 : !ct
}

// CHECK: test_rsub
func.func @test_rsub(%0: !ct, %1: !ct) -> !ct {
  // CHECK: lwe.rsub
  %2 = lwe.rsub %0, %1 : (!ct, !ct) -> !ct
  return %2 : !ct
}

// CHECK: test_rmul
func.func @test_rmul(%0: !ct, %1: !ct) -> !post_mul_ct {
  // CHECK: lwe.rmul
  %2 = lwe.rmul %0, %1 : (!ct, !ct) -> !post_mul_ct
  return %2 : !post_mul_ct
}

// CHECK: test_rnegate
func.func @test_rnegate(%0: !ct) -> !ct {
  // CHECK: lwe.rnegate
  %1 = lwe.rnegate %0 : !ct
  return %1 : !ct
}

// CHECK: test_radd_plain
func.func @test_radd_plain(%0: !ct, %1: !pt) -> !ct {
  // CHECK: lwe.radd_plain
  %2 = lwe.radd_plain %0, %1 : (!ct, !pt) -> !ct
  // CHECK: lwe.radd_plain
  %3 = lwe.radd_plain %1, %0 : (!pt, !ct) -> !ct
  return %2 : !ct
}

// CHECK: test_rsub_plain
func.func @test_rsub_plain(%0: !ct, %1: !pt) -> !ct {
  // CHECK: lwe.rsub_plain
  %2 = lwe.rsub_plain %0, %1 : (!ct, !pt) -> !ct
  // CHECK: lwe.rsub_plain
  %3 = lwe.rsub_plain %1, %0 : (!pt, !ct) -> !ct
  return %2 : !ct
}

// CHECK: test_rmul_plain
func.func @test_rmul_plain(%0: !ct, %1: !pt) -> !ct {
  // CHECK: lwe.rmul_plain
  %2 = lwe.rmul_plain %0, %1 : (!ct, !pt) -> !ct
  // CHECK: lwe.rmul_plain
  %3 = lwe.rmul_plain %1, %0 : (!pt, !ct) -> !ct
  return %2 : !ct
}

!Z4 = !mod_arith.int<4 : i32>
!Z3 = !mod_arith.int<3 : i32>
!Z5 = !mod_arith.int<5 : i32>
!rns_src_even = !rns.rns<!Z4, !Z3>
!rns_tgt_odd = !rns.rns<!Z5>
#ring_src_even = #polynomial.ring<coefficientType = !rns_src_even, polynomialModulus = <1 + x**8>>
#ring_tgt_odd = #polynomial.ring<coefficientType = !rns_tgt_odd, polynomialModulus = <1 + x**8>>
!ringelt_src_even = !lwe.lwe_ring_elt<ring = #ring_src_even>
!ringelt_tgt_odd = !lwe.lwe_ring_elt<ring = #ring_tgt_odd>

// CHECK: test_convert_basis
func.func @test_convert_basis(%arg0: !ringelt_src_even) -> !ringelt_tgt_odd {
  // CHECK: lwe.convert_basis
  %0 = "lwe.convert_basis"(%arg0) {targetBasis = !rns_tgt_odd} : (!ringelt_src_even) -> !ringelt_tgt_odd
  return %0 : !ringelt_tgt_odd
}
