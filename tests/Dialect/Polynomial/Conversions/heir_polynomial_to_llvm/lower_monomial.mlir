// RUN: heir-opt --polynomial-to-mod-arith %s | FileCheck %s

#cycl_2048 = #polynomial.int_polynomial<1 + x**1024>
!coeff_ty = !mod_arith.int<65536:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#cycl_2048>

func.func @test_monomial() -> !polynomial.polynomial<ring=#ring> {
  // CHECK: %[[deg:.*]] = arith.constant 1023
  %deg = arith.constant 1023 : index
  // CHECK: %[[five:.*]] = mod_arith.constant 5
  %five = mod_arith.constant 5 : !coeff_ty
  // CHECK: %[[container:.*]] = arith.constant dense<0>
  // CHECK: %[[container_mod:.*]] = mod_arith.encapsulate %[[container]]
  // CHECK: tensor.insert %[[five]] into %[[container_mod]][%[[deg]]]
  %0 = polynomial.monomial %five, %deg : (!coeff_ty, index) -> !polynomial.polynomial<ring=#ring>
  return %0 : !polynomial.polynomial<ring=#ring>
}
