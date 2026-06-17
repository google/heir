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
// CHECK: .he_mul[
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

// CHECK: def test_gen_params(
// CHECK: "scaling_factor": 563019763943521
// CHECK: "public_key":
// CHECK: "secret_key":
// CHECK: "evaluation_key":
func.func @test_gen_params(
    %pk: !jaxiteword.public_key<>,
    %sk: !jaxiteword.private_key<>,
    %ek: !jaxiteword.eval_key<>) -> !jaxiteword.crypto_context<> {
  %ctx = jaxiteword.gen_params %pk, %sk, %ek {
    degree = 8192 : i64,
    numSlots = 4096 : i64,
    scalingFactor = 563019763943521.0 : f64,
    qTowers = array<i64: 1, 2>,
    pTowers = array<i64: 3>,
    batch = 1 : i32,
    r = 4 : i32,
    c = 4 : i32,
    dnum = 3 : i32,
    numEvalMult = 2 : i32,
    compositeDegree = 1 : i32
  } : (!jaxiteword.public_key<>, !jaxiteword.private_key<>, !jaxiteword.eval_key<>) -> !jaxiteword.crypto_context<>
  return %ctx : !jaxiteword.crypto_context<>
}
