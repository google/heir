// RUN: heir-opt --convert-elementwise-to-affine --full-loop-unroll --convert-tensor-to-scalars --polynomial-to-pisa %s

// FIXME: ADD FILECHECK

!coeff_ty = !mod_arith.int<33538049:i32>
!poly = !polynomial.polynomial<ring=<coefficientType=!coeff_ty, polynomialModulus=#polynomial.int_polynomial<1 + x**8192>>>

func.func @tensor(%arg0: tensor<2x!poly>, %arg1: tensor<2x!poly>) -> tensor<2x!poly> {
  %0 = polynomial.add %arg0, %arg1 : tensor<2x!poly>
  return %0 : tensor<2x!poly>
}


func.func @scalar(%arg0: !poly, %arg1: !poly, %arg2: !poly, %arg3: !poly) -> (!poly, !poly) {
  %0 = polynomial.add %arg0, %arg2 : !poly
  %1 = polynomial.add %arg1, %arg3 : !poly
  return %0, %1 : !poly, !poly
}
