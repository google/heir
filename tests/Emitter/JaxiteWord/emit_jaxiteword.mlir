// RUN: heir-translate --emit-jaxiteword %s | FileCheck %s

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 29>
#key = #lwe.key<>
#ring_f64_1_x8 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**8>>
#ring_i32_1_x8 = #polynomial.ring<coefficientType = i32, polynomialModulus = <1 + x**8>>
#ciphertext_space = #lwe.ciphertext_space<ring = #ring_i32_1_x8, encryption_type = mix>
#modulus_chain = #lwe.modulus_chain<elements = <1095233372161 : i64>, current = 0>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space, key = #key, modulus_chain = #modulus_chain>

// CHECK: def test_add(
func.func @test_add(%ctx: !jaxiteword.crypto_context<>, %ct1 : !ct_L1, %ct2 : !ct_L1) -> !ct_L1 {
  %out = jaxiteword.add %ctx, %ct1, %ct2 : (!jaxiteword.crypto_context<>, !ct_L1, !ct_L1) -> !ct_L1
  return %out : !ct_L1
}

// CHECK: def test_mul(
// CHECK: hemul(
func.func @test_mul(%ctx: !jaxiteword.crypto_context<>, %ct1 : !ct_L1, %ct2 : !ct_L1) -> !ct_L1 {
  %pk, %sk = jaxiteword.gen_keypair %ctx : (!jaxiteword.crypto_context<>) -> (!jaxiteword.public_key<>, !jaxiteword.private_key<>)
  %ek = jaxiteword.gen_mulkey %ctx, %sk : (!jaxiteword.crypto_context<>, !jaxiteword.private_key<>) -> !jaxiteword.eval_key<>
  %out = jaxiteword.mul %ctx, %ct1, %ct2, %ek : (!jaxiteword.crypto_context<>, !ct_L1, !ct_L1, !jaxiteword.eval_key<>) -> !ct_L1
  return %out : !ct_L1
}

// CHECK: def test_mul_no_relin(
// CHECK: hemul_no_relin(
func.func @test_mul_no_relin(%ctx: !jaxiteword.crypto_context<>, %ct1 : !ct_L1, %ct2 : !ct_L1) -> !ct_L1 {
  %out = jaxiteword.mul_no_relin %ctx, %ct1, %ct2 : (!jaxiteword.crypto_context<>, !ct_L1, !ct_L1) -> !ct_L1
  return %out : !ct_L1
}
