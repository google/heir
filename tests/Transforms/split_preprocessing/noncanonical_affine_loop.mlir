// RUN: heir-opt %s --split-preprocessing --affine-loop-normalize | FileCheck %s

// split-preprocessing uses enclosing loop induction variables as storage
// indices. A non-zero lower bound or non-unit step must therefore be normalized
// before splitting, so the storage is indexed by iteration number rather than
// by the original induction variable values.

!Z36028797017456641_i64 = !mod_arith.int<36028797017456641 : i64>
!Z35184371138561_i64 = !mod_arith.int<35184371138561 : i64>
!Z35184372121601_i64 = !mod_arith.int<35184372121601 : i64>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 0>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L2 = !rns.rns<!Z36028797017456641_i64, !Z35184371138561_i64, !Z35184372121601_i64>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>>

// CHECK:       func.func @noncanonical_affine_loop__preprocessing
// CHECK:         affine.for %[[I:.*]] = 0 to 2 {
// CHECK-NOT:       affine.apply
// CHECK:           preprocessing.store %{{.*}}, %{{.*}}[%[[I]]] site 0
// CHECK:         return

module attributes {backend.openfhe, ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797017456641, 35184371138561, 35184372121601], P = [1152921504607338497, 1152921504608747521], logDefaultScale = 45>, scheme.ckks} {
  func.func @noncanonical_affine_loop() {
    %cst = arith.constant dense<1.0> : tensor<1024xf32>
    affine.for %i = 1 to 7 step 3 {
      %pt = lwe.rlwe_encode %cst {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x1024} : tensor<1024xf32> -> !pt
    }
    return
  }
}
