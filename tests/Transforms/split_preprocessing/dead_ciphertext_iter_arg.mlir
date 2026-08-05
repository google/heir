// RUN: heir-opt %s --split-preprocessing | FileCheck %s

// A loop carries a ciphertext iter_arg (the running accumulator) and also
// encodes a plaintext per iteration. The ciphertext value is not part of the
// plaintext slice, so when the loop is cloned into the preprocessing function
// its iter_arg becomes dead. remove-dead-values only poisons (it cannot
// structurally strip) a dead affine.for iter_arg, so split-preprocessing must
// remove it itself -- otherwise a ub.poison-typed loop-carried value survives
// and later backend lowering can neither convert nor emit it.

!Z36028797017456641_i64 = !mod_arith.int<36028797017456641 : i64>
!Z35184371138561_i64 = !mod_arith.int<35184371138561 : i64>
!Z35184372121601_i64 = !mod_arith.int<35184372121601 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#key = #lwe.key<>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L2 = !rns.rns<!Z36028797017456641_i64, !Z35184371138561_i64, !Z35184372121601_i64>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>>
#ring_rns_L2_1_x1024 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2_1_x1024, encryption_type = mix>
!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #lwe.modulus_chain<elements = <36028797017456641 : i64, 35184371138561 : i64, 35184372121601 : i64>, current = 2>>

// The preprocessing loop must be a pure store loop: no iter_args, no ub.poison.
// CHECK:       func.func @f__preprocessing() -> !preprocessing.storage<!pt>
// CHECK-NOT:     ub.poison
// CHECK:         affine.for %[[I:.*]] = 0 to 4 {
// CHECK-NOT:     iter_args
// CHECK:           %[[PT:.*]] = lwe.rlwe_encode
// CHECK:           preprocessing.store %[[PT]], %{{.*}}[%[[I]]] site 0<!pt> : !pt, <!pt>
// CHECK:         return

module attributes {backend.openfhe, ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797017456641, 35184371138561, 35184372121601], P = [1152921504607338497, 1152921504608747521], logDefaultScale = 45>, scheme.ckks} {
  func.func @f(%arg0: tensor<1x!ct_L2>) -> tensor<1x!ct_L2> {
    %cst = arith.constant dense<1.0> : tensor<1024xf32>
    %0 = affine.for %i = 0 to 4 iter_args(%sum = %arg0) -> (tensor<1x!ct_L2>) {
      %pt = lwe.rlwe_encode %cst {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
      %from = tensor.from_elements %pt : tensor<1x!pt>
      %1 = ckks.add_plain %sum, %from : (tensor<1x!ct_L2>, tensor<1x!pt>) -> tensor<1x!ct_L2>
      affine.yield %1 : tensor<1x!ct_L2>
    }
    return %0 : tensor<1x!ct_L2>
  }
}
