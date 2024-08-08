// RUN: heir-opt --expand-rns %s | FileCheck  %s

!poly1 = !polynomial.polynomial<ring=<coefficientType = i32, coefficientModulus = 1073479681 : i32, polynomialModulus=#polynomial.int_polynomial<1 + x**8192>>>
!poly2 = !polynomial.polynomial<ring=<coefficientType = i32, coefficientModulus = 1071513601 : i32, polynomialModulus=#polynomial.int_polynomial<1 + x**8192>>>
!rns = !polynomial.rns<!poly1,!poly2>

func.func @test_rns(%arg0: !rns, %arg1: !rns) ->  !rns {
  %0 = polynomial.add %arg0, %arg1 : !rns
  return %0 :  !rns
}

// // For this to work, you need the "detensorize" PR/branch, which adds `--convert-tensor-to-scalars`.
// // With that, use --convert-elementwise-to-affine --full-loop-unroll --convert-tensor-to-scalars --expand-rns
// func.func @test_elementwise_rns(%arg0: tensor<2x!rns>, %arg1: tensor<2x!rns>) ->  tensor<2x!rns> {
//   %0 = polynomial.add %arg0, %arg1 : tensor<2x!rns>
//   return %0 :  tensor<2x!rns>
// }
