!Z0 = !mod_arith.int<17 : i64>
!Z1 = !mod_arith.int<97 : i64>
!rns = !rns.rns<!Z0, !Z1>
#ring = #polynomial.ring<coefficientType = !rns, polynomialModulus = <1 + x**8>>
!poly = !polynomial.polynomial<ring = #ring>

#v0 = #mod_arith.value<3 : !Z0>
#v1 = #mod_arith.value<8 : !Z1>
#rns_root = #rns.value<[#v0, #v1]>
#root = #polynomial.primitive_root<value = #rns_root, degree = 16 : i32>

!ntt_poly = !polynomial.polynomial<ring = #ring, form = eval>

func.func public @test_ntt_rns() -> !ntt_poly {
  %coeffsRaw = arith.constant dense<[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8]]> : tensor<8x2xi64>
  %coeffs = mod_arith.encapsulate %coeffsRaw : tensor<8x2xi64> -> tensor<8x!rns>
  %poly = polynomial.from_tensor %coeffs : tensor<8x!rns> -> !poly
  %res = polynomial.ntt %poly {root = #root} : !poly
  return %res : !ntt_poly
}
