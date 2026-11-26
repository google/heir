// RUN: heir-opt --lwe-to-polynomial %s | FileCheck %s

// Small moduli, so that P = 97 * 113 = 10961 is reduced modulo each q_j:
//   10961 mod 17 = 13, 10961 mod 41 = 14, 10961 mod 73 = 11.
!Zq0 = !mod_arith.int<17 : i32>
!Zq1 = !mod_arith.int<41 : i32>
!Zq2 = !mod_arith.int<73 : i32>
!Zp0 = !mod_arith.int<97 : i32>
!Zp1 = !mod_arith.int<113 : i32>

!rns_Q = !rns.rns<!Zq0, !Zq1, !Zq2>
!rns_QP = !rns.rns<!Zq0, !Zq1, !Zq2, !Zp0, !Zp1>

#ring_Q = #polynomial.ring<coefficientType = !rns_Q, polynomialModulus = <1 + x**8>>
#ring_QP = #polynomial.ring<coefficientType = !rns_QP, polynomialModulus = <1 + x**8>>

#encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>

!sk = !lwe.lwe_secret_key<key = #key, ring = #ring_Q>
!pk = !lwe.lwe_public_key<key = #key, ring = #ring_Q>
!ksk_ct = !lwe.lwe_ciphertext<encoding = #encoding, ciphertext_space = <ring = #ring_QP, encryption_type = ckks>, key = #key>

// CHECK: @test_gen_secret_key
// CHECK-SAME: -> tensor<1x[[POLY_Q:.*]]>
func.func @test_gen_secret_key() -> !sk {
  // CHECK: %[[PRNG:.*]] = random.init_prng
  // CHECK: %[[TERNARY:.*]] = random.discrete_uniform_distribution %[[PRNG]]{range = [-1 : i32, 2 : i32]}
  // CHECK: %[[S:.*]] = random.sample %[[TERNARY]] {{.*}} -> [[POLY_Q]]
  // CHECK: %[[SK:.*]] = tensor.from_elements %[[S]] : tensor<1x[[POLY_Q]]>
  // CHECK: return %[[SK]]
  %sk = lwe.gen_secret_key : () -> !sk
  return %sk : !sk
}

// CHECK: @test_gen_public_key
// CHECK-SAME: (%[[SK:.*]]: tensor<1x[[POLY_Q:.*]]>) -> tensor<2x[[POLY_Q]]>
func.func @test_gen_public_key(%sk: !sk) -> !pk {
  // CHECK: %[[PRNG:.*]] = random.init_prng
  // CHECK: %[[GAUSSIAN:.*]] = random.discrete_gaussian_distribution %[[PRNG]]
  // CHECK-NOT: random.discrete_uniform_distribution
  // CHECK: %[[S:.*]] = tensor.extract %[[SK]]
  // CHECK: %[[A_COEFFS:.*]] = random.sample_modq %[[PRNG]]
  // CHECK: %[[A:.*]] = polynomial.from_tensor %[[A_COEFFS]]
  // CHECK: %[[E:.*]] = random.sample %[[GAUSSIAN]]
  // CHECK-NOT: polynomial.mul_scalar
  // CHECK: %[[AS:.*]] = polynomial.mul %[[A]], %[[S]]
  // CHECK: %[[B:.*]] = polynomial.sub %[[E]], %[[AS]]
  // CHECK: %[[PK:.*]] = tensor.from_elements %[[B]], %[[A]] : tensor<2x[[POLY_Q]]>
  // CHECK: return %[[PK]]
  %pk = lwe.gen_public_key %sk : (!sk) -> !pk
  return %pk : !pk
}

// For BGV, the noise is scaled by t = 65537, which is
//   65537 mod 17 = 2, 65537 mod 41 = 19, 65537 mod 73 = 56.
// CHECK: @test_gen_public_key_bgv
func.func @test_gen_public_key_bgv(%sk: !sk) -> !pk {
  // CHECK: %[[T:.*]] = rns.constant <[#mod_arith.value<2 : {{.*}}, #mod_arith.value<19 : {{.*}}, #mod_arith.value<56 : {{.*}}]>
  // CHECK: %[[E:.*]] = random.sample %
  // CHECK: %[[SCALED_E:.*]] = polynomial.mul_scalar %[[E]], %[[T]]
  // CHECK: %[[AS:.*]] = polynomial.mul
  // CHECK: polynomial.sub %[[SCALED_E]], %[[AS]]
  %pk = lwe.gen_public_key %sk <plaintext_modulus = 65537> : (!sk) -> !pk
  return %pk : !pk
}

// Q has 3 limbs and P has 2, so Q is split into the digits (q0, q1) and (q2).
// CHECK: @test_gen_relin_key
// CHECK-SAME: (%[[SK:.*]]: tensor<1x[[POLY_Q:.*]]>) -> tensor<2x2x[[POLY_QP:.*]]>
func.func @test_gen_relin_key(%sk: !sk) -> tensor<2x!ksk_ct> {
  // CHECK: %[[PRNG:.*]] = random.init_prng
  // CHECK: %[[GAUSSIAN:.*]] = random.discrete_gaussian_distribution %[[PRNG]]
  // CHECK: %[[S_Q:.*]] = tensor.extract %[[SK]]

  // The ternary secret key is lifted exactly from Q to QP, via a centered lift
  // of its first residue.
  // CHECK: %[[S:.*]] = polynomial.apply_coefficientwise(%[[S_Q]] : [[POLY_Q]])
  // CHECK: rns.extract_residue {{.*}} <index = 0>
  // CHECK: mod_arith.lift centered
  // CHECK-COUNT-5: mod_arith.encapsulate
  // CHECK: rns.pack
  // CHECK: -> [[POLY_QP]]
  // CHECK: %[[S_SQ:.*]] = polynomial.mul %[[S]], %[[S]] : [[POLY_QP]]

  // Digit 0: P * g_0 = (P mod q0, P mod q1, 0, 0, 0)
  // CHECK: %[[G0:.*]] = rns.constant <[#mod_arith.value<13 : {{.*}}, #mod_arith.value<14 : {{.*}}, #mod_arith.value<0 : {{.*}}, #mod_arith.value<0 : {{.*}}, #mod_arith.value<0 : {{.*}}]>
  // CHECK: %[[M0:.*]] = polynomial.mul_scalar %[[S_SQ]], %[[G0]]
  // CHECK: %[[A0_COEFFS:.*]] = random.sample_modq %[[PRNG]]
  // CHECK: %[[A0:.*]] = polynomial.from_tensor %[[A0_COEFFS]]
  // CHECK: %[[E0:.*]] = random.sample %[[GAUSSIAN]]
  // CHECK: %[[E0_M0:.*]] = polynomial.add %[[E0]], %[[M0]]
  // CHECK: %[[A0S:.*]] = polynomial.mul %[[A0]], %[[S]]
  // CHECK: %[[B0:.*]] = polynomial.sub %[[E0_M0]], %[[A0S]]

  // Digit 1: P * g_1 = (0, 0, P mod q2, 0, 0)
  // CHECK: %[[G1:.*]] = rns.constant <[#mod_arith.value<0 : {{.*}}, #mod_arith.value<0 : {{.*}}, #mod_arith.value<11 : {{.*}}, #mod_arith.value<0 : {{.*}}, #mod_arith.value<0 : {{.*}}]>
  // CHECK: %[[M1:.*]] = polynomial.mul_scalar %[[S_SQ]], %[[G1]]
  // CHECK: %[[A1_COEFFS:.*]] = random.sample_modq %[[PRNG]]
  // CHECK: %[[A1:.*]] = polynomial.from_tensor %[[A1_COEFFS]]
  // CHECK: %[[E1:.*]] = random.sample %[[GAUSSIAN]]
  // CHECK: %[[E1_M1:.*]] = polynomial.add %[[E1]], %[[M1]]
  // CHECK: %[[A1S:.*]] = polynomial.mul %[[A1]], %[[S]]
  // CHECK: %[[B1:.*]] = polynomial.sub %[[E1_M1]], %[[A1S]]

  // CHECK: %[[RLK:.*]] = tensor.from_elements %[[B0]], %[[A0]], %[[B1]], %[[A1]] : tensor<2x2x[[POLY_QP]]>
  // CHECK: return %[[RLK]]
  %rlk = lwe.gen_relin_key %sk : (!sk) -> tensor<2x!ksk_ct>
  return %rlk : tensor<2x!ksk_ct>
}
