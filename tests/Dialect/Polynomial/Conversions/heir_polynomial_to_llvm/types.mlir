// RUN: heir-opt --polynomial-to-mod-arith %s | FileCheck %s

#cycl_2048 = #polynomial.int_polynomial<1 + x**1024>
!coeff_ty = !mod_arith.int<65536:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#cycl_2048>
module {
  // CHECK-label: f0
  // CHECK %arg0: tensor<1024xui64, #polynomial.ring<coefficientType=!mod_arith.int<4294967296:i32>, polynomialModulus=#polynomial.int_polynomial<1 + x**1024>>>
  func.func @f0(%arg: !polynomial.polynomial<ring=#ring>) -> !polynomial.polynomial<ring=#ring> {
    return %arg : !polynomial.polynomial<ring=#ring>
  }

  // CHECK-label: test_lower_fn_and_call
  // CHECK-NOT: polynomial.polynomial<#ring>
  func.func @test_lower_fn_and_call(%arg: !polynomial.polynomial<ring=#ring>) -> !polynomial.polynomial<ring=#ring>
 {
    %0 = func.call @f0(%arg) : (!polynomial.polynomial<ring=#ring>) -> !polynomial.polynomial<ring=#ring>
    return %0 : !polynomial.polynomial<ring=#ring>
  }
}
