// RUN: heir-opt --lwe-to-openfhe %s | FileCheck %s

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <36028797005856769 : i64, 35184478519297 : i64>, current = 1>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L45 = !rns.rns<!mod_arith.int<36028797005856769 : i64>, !mod_arith.int<35184478519297 : i64>>
#ring_rns_L45_1_x1024 = #polynomial.ring<coefficientType = !rns_L45, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L45 = #lwe.ciphertext_space<ring = #ring_rns_L45_1_x1024, encryption_type = mix>
!ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L45, key = #key, modulus_chain = #modulus_chain>

// CHECK: ![[CC:.*]] = !openfhe.crypto_context
// CHECK: ![[CT:.*]] = !openfhe.ciphertext

module attributes {backend.openfhe, ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797005856769, 35184478519297], P = [1152921504616808449], logDefaultScale = 45>, scheme.ckks} {
  // CHECK: func.func @test_eval_chebyshev(%[[VAL_0:.*]]: ![[CC]], %[[VAL_1:.*]]: ![[CT]]) -> ![[CT]]
  // CHECK: %[[VAL_2:.*]] = openfhe.eval_chebyshev_series %[[VAL_0]], %[[VAL_1]] {coefficients = [2.000000e+00, 2.000000e+00], domain_lower = -1.000000e+00 : f64, domain_upper = 1.000000e+00 : f64} : (![[CC]], ![[CT]]) -> ![[CT]]
  // CHECK: return %[[VAL_2]] : ![[CT]]
  func.func @test_eval_chebyshev(%ct: !ct) -> !ct {
    %0 = kernel.eval_chebyshev %ct {coefficients = [1.0 : f64, 2.0 : f64]} : !ct -> !ct
    return %0 : !ct
  }
}
