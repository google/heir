module {
  func.func @basic_test(%arg0: tensor<2x!polynomial.polynomial<ring = <coefficientType = i32, coefficientModulus = 463187969 : i32, polynomialModulus = <1 + x**8192>>>>, %arg1: tensor<2x!polynomial.polynomial<ring = <coefficientType = i32, coefficientModulus = 463187969 : i32, polynomialModulus = <1 + x**8192>>>>) -> tensor<2x!polynomial.polynomial<ring = <coefficientType = i32, coefficientModulus = 463187969 : i32, polynomialModulus = <1 + x**8192>>>> {
    %0 = polynomial.add %arg0, %arg1 : tensor<2x!polynomial.polynomial<ring = <coefficientType = i32, coefficientModulus = 463187969 : i32, polynomialModulus = <1 + x**8192>>>>
    return %0 : tensor<2x!polynomial.polynomial<ring = <coefficientType = i32, coefficientModulus = 463187969 : i32, polynomialModulus = <1 + x**8192>>>>
  }
}
