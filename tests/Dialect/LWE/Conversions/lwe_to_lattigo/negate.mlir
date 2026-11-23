// RUN: heir-opt --lwe-to-lattigo --split-input-file %s | FileCheck %s

!Z35184372121601_i64 = !mod_arith.int<35184372121601 : i64>
!Z36028797018652673_i64 = !mod_arith.int<36028797018652673 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#modulus_chain_L1_C1 = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 1>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L1 = !rns.rns<!Z36028797018652673_i64, !Z35184372121601_i64>
#ring_rns_L1_1_x1024 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024, encryption_type = ckks>
!ct_L1 = !lwe.lwe_ciphertext<encoding = #inverse_canonical_encoding, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L1_C1>

// CHECK: func.func @negate(%[[ev:.*]]: !evaluator, {{.*}}, %[[ct:.*]]: !ct)
// CHECK: %[[neg:.*]] = lattigo.rlwe.negate_new %[[ev]], %[[ct]]
// CHECK: return %[[neg]]
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 35184372121601], P = [1152921504606994433], logDefaultScale = 45>, scheme.ckks} {
  func.func @negate(%arg0: !ct_L1) -> !ct_L1 {
    %0 = lwe.rnegate %arg0 : !ct_L1
    return %0 : !ct_L1
  }
}
