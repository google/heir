// RUN: heir-translate --emit-jaxiteword %s | FileCheck %s

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 29>
#key = #lwe.key<>
#ring_f64_1_x8 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**8>>
#ring_i32_1_x8 = #polynomial.ring<coefficientType = i32, polynomialModulus = <1 + x**8>>
#ciphertext_space = #lwe.ciphertext_space<ring = #ring_i32_1_x8, encryption_type = mix>
#ciphertext_space_D3 = #lwe.ciphertext_space<ring = #ring_i32_1_x8, encryption_type = mix, size = 3>
#modulus_chain = #lwe.modulus_chain<elements = <1095233372161 : i64>, current = 0>
#modulus_chain_L2 = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64>, current = 1>
!ct_L1 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space, key = #key, modulus_chain = #modulus_chain>
!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space, key = #key, modulus_chain = #modulus_chain_L2>
!ct_L2_D3 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_D3, key = #key, modulus_chain = #modulus_chain_L2>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x8, encoding = #inverse_canonical_encoding>>

// CHECK: def test_add(
// CHECK: {{.*}}: Polynomial,
// CHECK: .he_add[
// CHECK-SAME: ].add(
func.func @test_add(%ctx: !jaxiteword.crypto_context<>, %ct1 : !ct_L1, %ct2 : !ct_L1) -> !ct_L1 {
  %out = jaxiteword.add %ctx, %ct1, %ct2 : (!jaxiteword.crypto_context<>, !ct_L1, !ct_L1) -> !ct_L1
  return %out : !ct_L1
}

// CHECK: def test_add_plain(
// CHECK: {{.*}} = {{.*}}.he_add[{{.*}}.max_level].add_plain({{.*}}, {{.*}})
func.func @test_add_plain(%ctx: !jaxiteword.crypto_context<>, %ct : !ct_L2, %pt : !pt) -> !ct_L2 {
  %out = jaxiteword.add_plain %ctx, %ct, %pt : (!jaxiteword.crypto_context<>, !ct_L2, !pt) -> !ct_L2
  return %out : !ct_L2
}

// CHECK: def test_sub(
// CHECK: .he_sub[
// CHECK-SAME: ].sub(
func.func @test_sub(%ctx: !jaxiteword.crypto_context<>, %ct1 : !ct_L1, %ct2 : !ct_L1) -> !ct_L1 {
  %out = jaxiteword.sub %ctx, %ct1, %ct2 : (!jaxiteword.crypto_context<>, !ct_L1, !ct_L1) -> !ct_L1
  return %out : !ct_L1
}

// CHECK: def test_mul(
// CHECK: {{.*}}_raw = key_gen.gen_evaluation_key
// CHECK: {{.*}} = [
// CHECK: jnp.array({{.*}}_raw["a"], dtype=jnp.uint32).transpose(0, 2, 1),
// CHECK: jnp.array({{.*}}_raw["b"], dtype=jnp.uint32).transpose(0, 2, 1),
// CHECK: .he_mul[
func.func @test_mul(%ctx: !jaxiteword.crypto_context<>, %ct1 : !ct_L1, %ct2 : !ct_L1) -> !ct_L1 {
  %pk, %sk = jaxiteword.gen_keypair %ctx : (!jaxiteword.crypto_context<>) -> (!jaxiteword.public_key<>, !jaxiteword.private_key<>)
  %ek = jaxiteword.gen_mulkey %ctx, %sk : (!jaxiteword.crypto_context<>, !jaxiteword.private_key<>) -> !jaxiteword.eval_key<>
  %out = jaxiteword.mul %ctx, %ct1, %ct2, %ek : (!jaxiteword.crypto_context<>, !ct_L1, !ct_L1, !jaxiteword.eval_key<>) -> !ct_L1
  return %out : !ct_L1
}

// CHECK: def test_mul_no_relin(
// CHECK: .hemul_no_relin(
func.func @test_mul_no_relin(%ctx: !jaxiteword.crypto_context<>, %ct1 : !ct_L1, %ct2 : !ct_L1) -> !ct_L1 {
  %out = jaxiteword.mul_no_relin %ctx, %ct1, %ct2 : (!jaxiteword.crypto_context<>, !ct_L1, !ct_L1) -> !ct_L1
  return %out : !ct_L1
}

// CHECK: def test_relin(
// CHECK: .relinearize(
func.func @test_relin(%ctx: !jaxiteword.crypto_context<>, %ct: !ct_L2_D3, %ek: !jaxiteword.eval_key<>) -> !ct_L2 {
  %out = jaxiteword.relin %ctx, %ct, %ek : (!jaxiteword.crypto_context<>, !ct_L2_D3, !jaxiteword.eval_key<>) -> !ct_L2
  return %out : !ct_L2
}

// CHECK: def test_rescale(
// CHECK: .he_rescale[
// CHECK-SAME: ].rescale(
func.func @test_rescale(%ctx: !jaxiteword.crypto_context<>, %ct: !ct_L2) -> !ct_L1 {
  %out = jaxiteword.mod_reduce %ctx, %ct : (!jaxiteword.crypto_context<>, !ct_L2) -> !ct_L1
  return %out : !ct_L1
}

// CHECK: def test_rotate(
// CHECK: .he_rot[
// CHECK-SAME: ].rotate(
func.func @test_rotate(%ctx: !jaxiteword.crypto_context<>, %ct: !ct_L1, %ek: !jaxiteword.eval_key<>) -> !ct_L1 {
  %out = jaxiteword.rot %ctx, %ct, %ek {index = 2 : i64} : (!jaxiteword.crypto_context<>, !ct_L1, !jaxiteword.eval_key<>) -> !ct_L1
  return %out : !ct_L1
}

// CHECK: def test_mul_plain(
// CHECK: .ptct_mul[
// CHECK-SAME: ].mul({{.*}}, {{.*}})
func.func @test_mul_plain(%ctx: !jaxiteword.crypto_context<>, %ct: !ct_L1, %pt: !pt) -> !ct_L1 {
  %out = jaxiteword.mul_plain %ctx, %ct, %pt : (!jaxiteword.crypto_context<>, !ct_L1, !pt) -> !ct_L1
  return %out : !ct_L1
}

// CHECK: def test_floor_div_si(
// CHECK: {{.*}} = {{.*}} // {{.*}}
func.func @test_floor_div_si(%lhs: i32, %rhs: i32) -> i32 {
  %out = arith.floordivsi %lhs, %rhs : i32
  return %out : i32
}

// CHECK: def test_gen_params(
// CHECK: params = {
// CHECK: "scaling_factor": 563019763943521
// CHECK: "output_scale": 563019763943521
// CHECK-NOT: "public_key":
// CHECK-NOT: "secret_key":
// CHECK-NOT: "evaluation_key":
// CHECK: {{.*}} = ckks.CKKSContext(params)
func.func @test_gen_params() -> !jaxiteword.crypto_context<> {
  %ctx = jaxiteword.gen_params {
    degree = 8192 : i64,
    numSlots = 4096 : i64,
    scalingFactor = 563019763943521.0 : f64,
    qTowers = array<i64: 1, 2>,
    pTowers = array<i64: 3>,
    batch = 1 : i32,
    r = 4 : i32,
    c = 4 : i32,
    dnum = 3 : i32,
    compositeDegree = 1 : i32
  } : () -> !jaxiteword.crypto_context<>
  return %ctx : !jaxiteword.crypto_context<>
}

// CHECK: def test_program_initialization(
// CHECK: {{.*}}.public_key = {{.*}}
// CHECK: {{.*}}.secret_key = {{.*}}
// CHECK: {{.*}}.evaluation_key = {{.*}}
// CHECK: {{.*}}.parameters["public_key"] = {{.*}}
// CHECK: {{.*}}.parameters["secret_key"] = {{.*}}
// CHECK: {{.*}}.parameters["evaluation_key"] = {{.*}}
// CHECK: {{.*}}.program_initialization(total_rotation_indices=[1, 2], dnum=3, r=4, c=4, batch=1)
func.func @test_program_initialization(
    %ctx: !jaxiteword.crypto_context<>,
    %pk: !jaxiteword.public_key<>,
    %sk: !jaxiteword.private_key<>,
    %ek: !jaxiteword.eval_key<>) {
  jaxiteword.program_initialization %ctx, %pk, %sk, %ek {
    totalRotationIndices = array<i64: 1, 2>,
    dnum = 3 : i32,
    r = 4 : i32,
    c = 4 : i32,
    batch = 1 : i32
  } : (!jaxiteword.crypto_context<>, !jaxiteword.public_key<>, !jaxiteword.private_key<>, !jaxiteword.eval_key<>) -> ()
  return
}
