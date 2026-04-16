!Z0 = !mod_arith.int<17 : i64>
!Z1 = !mod_arith.int<97 : i64>
!rns = !rns.rns<!Z0, !Z1>
#ring = #polynomial.ring<coefficientType = !rns, polynomialModulus = <1 + x**8>>
!poly = !polynomial.polynomial<ring = #ring>

#v0 = #mod_arith.value<6 : !Z0>
#v1 = #mod_arith.value<85 : !Z1>
#rns_root = #rns.value<[#v0, #v1]>
#root = #polynomial.primitive_root<value = #rns_root, degree = 16 : i32>

!ntt_poly = !polynomial.polynomial<ring = #ring, form = eval>

func.func public @test_intt_rns() -> tensor<8x!rns> {
  %coeffsRaw = arith.constant dense<[[5, 86], [9, 4], [13, 41], [5, 53], [0, 56], [11, 4], [8, 67], [8, 85]]> : tensor<8x2xi64>
  %coeffs = mod_arith.encapsulate %coeffsRaw : tensor<8x2xi64> -> tensor<8x!rns>
  %poly = polynomial.from_tensor %coeffs : tensor<8x!rns> -> !ntt_poly
  %res = polynomial.intt %poly {root = #root} : !ntt_poly
  %res_tensor = polynomial.to_tensor %res : !poly -> tensor<8x!rns>
  return %res_tensor : tensor<8x!rns>
}
