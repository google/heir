// RUN: heir-opt %s --split-preprocessing='max-return-values=16' | FileCheck %s
// TODO(#2960): remove XFAIL and update test appropriately
// XFAIL: *

// CHECK: func.func @if_else_encode__preprocessing

!Z35184371138561_i64 = !mod_arith.int<35184371138561 : i64>
!Z35184372121601_i64 = !mod_arith.int<35184372121601 : i64>
!Z36028797017456641_i64 = !mod_arith.int<36028797017456641 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L2 = !rns.rns<!Z36028797017456641_i64, !Z35184371138561_i64, !Z35184372121601_i64>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>>
#ring_rns_L2_1_x1024 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2_1_x1024, encryption_type = mix>
!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #lwe.modulus_chain<elements = <36028797017456641 : i64, 35184371138561 : i64, 35184372121601 : i64>, current = 2>>

module attributes {backend.openfhe, ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797017456641, 35184371138561, 35184372121601], P = [1152921504607338497, 1152921504608747521], logDefaultScale = 45>, scheme.ckks} {
  func.func @if_else_encode(%arg0: tensor<1x!ct_L2>, %cond: i1) -> tensor<1x!ct_L2> {
    %cst1 = arith.constant dense<1.0> : tensor<1024xf32>
    %cst2 = arith.constant dense<2.0> : tensor<1024xf32>

    %pt = scf.if %cond -> (!pt) {
      %pt1 = lwe.rlwe_encode %cst1 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
      scf.yield %pt1 : !pt
    } else {
      %pt2 = lwe.rlwe_encode %cst2 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
      scf.yield %pt2 : !pt
    }

    %from_elements = tensor.from_elements %pt : tensor<1x!pt>
    %1 = ckks.add_plain %from_elements, %arg0 : (tensor<1x!pt>, tensor<1x!ct_L2>) -> tensor<1x!ct_L2>

    return %1 : tensor<1x!ct_L2>
  }
}
