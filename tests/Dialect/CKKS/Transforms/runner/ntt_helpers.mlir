// NTT conversions used by the runner tests to move data in and out of the
// lowered test functions, which take and return some values in NTT (eval)
// form. Each helper uses the same ring (and hence the same roots from
// --attach-ntt-roots) as relinearize.mlir and mul_relinearize.mlir.

!Zq0 = !mod_arith.int<1095233372161 : i64>
!Zq1 = !mod_arith.int<1032955396097 : i64>
!Zp0 = !mod_arith.int<998595133441 : i64>

!rns_Q = !rns.rns<!Zq0, !Zq1>
!rns_QP = !rns.rns<!Zq0, !Zq1, !Zp0>

#ring_Q = #polynomial.ring<coefficientType = !rns_Q, polynomialModulus = <1 + x**1024>>
#ring_QP = #polynomial.ring<coefficientType = !rns_QP, polynomialModulus = <1 + x**1024>>

!poly_Q = !polynomial.polynomial<ring = #ring_Q>
!ntt_poly_Q = !polynomial.polynomial<ring = #ring_Q, form = eval>
!poly_QP = !polynomial.polynomial<ring = #ring_QP>
!ntt_poly_QP = !polynomial.polynomial<ring = #ring_QP, form = eval>

// Converts one polynomial of the relinearization key to NTT form.
func.func public @key_to_ntt(%arg0: tensor<1024x!rns_QP>) -> !ntt_poly_QP {
  %0 = polynomial.from_tensor %arg0 : tensor<1024x!rns_QP> -> !poly_QP
  %1 = polynomial.ntt %0 : !poly_QP
  return %1 : !ntt_poly_QP
}

// Converts one ciphertext component to NTT form.
func.func public @ct_to_ntt(%arg0: tensor<1024x!rns_Q>) -> !ntt_poly_Q {
  %0 = polynomial.from_tensor %arg0 : tensor<1024x!rns_Q> -> !poly_Q
  %1 = polynomial.ntt %0 : !poly_Q
  return %1 : !ntt_poly_Q
}

// Converts one ciphertext component from NTT form back to coefficient form.
func.func public @ct_from_ntt(%arg0: !ntt_poly_Q) -> !poly_Q {
  %0 = polynomial.intt %arg0 : !ntt_poly_Q
  return %0 : !poly_Q
}
