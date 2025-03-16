// RUN: heir-opt %s --lower-polynomial-eval | FileCheck %s

#ring_i32_ = #polynomial.ring<coefficientType = i32>
!poly = !polynomial.polynomial<ring = #ring_i32_>
!Z65537 = !mod_arith.int<65537 : i64>

func.func @test_mod_arith_coeffs(%arg0: !Z65537) -> !Z65537 {
  %0 = polynomial.eval #polynomial<typed_int_polynomial <1 + 2x + 3x**2 + 4x**3> : !poly>, %arg0 : !Z65537
  return %0 : !Z65537
}
