// Converts one polynomial of the relinearization key to NTT form, using the same
// ring (and hence the same roots from --attach-ntt-roots) as relinearize.mlir.

!Zq0 = !mod_arith.int<1095233372161 : i64>
!Zq1 = !mod_arith.int<1032955396097 : i64>
!Zp0 = !mod_arith.int<998595133441 : i64>

!rns_QP = !rns.rns<!Zq0, !Zq1, !Zp0>

#ring_QP = #polynomial.ring<coefficientType = !rns_QP, polynomialModulus = <1 + x**1024>>

!poly = !polynomial.polynomial<ring = #ring_QP>
!ntt_poly = !polynomial.polynomial<ring = #ring_QP, form = eval>

func.func public @key_to_ntt(%arg0: tensor<1024x!rns_QP>) -> !ntt_poly {
  %0 = polynomial.from_tensor %arg0 : tensor<1024x!rns_QP> -> !poly
  %1 = polynomial.ntt %0 : !poly
  return %1 : !ntt_poly
}
