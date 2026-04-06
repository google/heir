// RUN: heir-opt --polynomial-to-mod-arith --verify-diagnostics %s

#ring_i256 = #polynomial.ring<coefficientType = i256>
!poly = !polynomial.polynomial<ring = #ring_i256>

func.func @constraints() -> !poly {
  %c2 = arith.constant 2 : index
  %c256_i256 = arith.constant 256 : i256
  %0 = polynomial.monomial %c256_i256, %c2 : (i256, index) -> !poly
  // expected-error@+1 {{polynomial-to-mod-arith requires all polynomial types have a polynomialModulus attribute}}
  return %0 : !poly
}
