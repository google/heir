// RUN: heir-opt %s > %t

// This simply tests for syntax.

#my_poly = #polynomial.int_polynomial<1 + x**1024>
#ring1 = #polynomial.ring<coefficientType = i32, coefficientModulus = 7917 : i32, polynomialModulus=#my_poly>
#ring2 = #polynomial.ring<coefficientType = i32, coefficientModulus = 3123 : i32, polynomialModulus=#my_poly>

module {
  func.func @test_ops(%p0 : !polynomial.polynomial<ring=#ring1>) {
    %cmod_switch = poly_ext.cmod_switch %p0 {congruence_modulus=117 : i16} : !polynomial.polynomial<ring=#ring1> -> !polynomial.polynomial<ring=#ring2>
    return
  }

  func.func @test_elementwise_ops(%p0 : !polynomial.polynomial<ring=#ring1>, %p1: !polynomial.polynomial<ring=#ring1>) {
    %tp0 = tensor.from_elements %p0, %p1 : tensor<2x!polynomial.polynomial<ring=#ring1>>

    %cmod_switch = poly_ext.cmod_switch %tp0 {congruence_modulus=117} : tensor<2x!polynomial.polynomial<ring=#ring1>> -> tensor<2x!polynomial.polynomial<ring=#ring2>>

    return
  }
}
