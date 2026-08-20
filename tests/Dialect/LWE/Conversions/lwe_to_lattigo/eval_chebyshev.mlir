// RUN: heir-opt --lwe-to-lattigo %s | FileCheck %s

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0 = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = mix>
!ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain>

// CHECK: ![[CT:.*]] = !lattigo.rlwe.ciphertext
// CHECK: ![[ENCODER:.*]] = !lattigo.ckks.encoder
// CHECK: ![[EVAL:.*]] = !lattigo.ckks.evaluator
// CHECK: ![[PARAM:.*]] = !lattigo.ckks.parameter
// CHECK: ![[POLY_EVAL:.*]] = !lattigo.ckks.polynomial_evaluator

module attributes {backend.lattigo, ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 35184372121601], P = [1152921504606994433], logDefaultScale = 45, encryptionTechnique = extended>, scheme.ckks} {
  // CHECK: func.func @test_eval_chebyshev(%[[VAL_0:.*]]: ![[EVAL]], %[[VAL_1:.*]]: ![[PARAM]], %[[VAL_2:.*]]: ![[ENCODER]], %[[VAL_3:.*]]: ![[CT]]{{.*}}) -> ![[CT]]
  // CHECK: %[[VAL_4:.*]] = lattigo.ckks.new_polynomial_evaluator %[[VAL_1]], %[[VAL_0]] : (![[PARAM]], ![[EVAL]]) -> ![[POLY_EVAL]]
  // CHECK: %[[VAL_5:.*]] = lattigo.ckks.chebyshev %[[VAL_4]], %[[VAL_3]] {coefficients = [1.000000e+00, 2.000000e+00], domain = array<f64: -1.000000e+00, 1.000000e+00>, targetScale = 35184372088832 : i64} : (![[POLY_EVAL]], ![[CT]]) -> ![[CT]]
  // CHECK: return %[[VAL_5]] : ![[CT]]
  func.func @test_eval_chebyshev(%ct: !ct) -> !ct {
    %0 = kernel.eval_chebyshev %ct {coefficients = [1.0 : f64, 2.0 : f64]} : !ct -> !ct
    return %0 : !ct
  }
}
