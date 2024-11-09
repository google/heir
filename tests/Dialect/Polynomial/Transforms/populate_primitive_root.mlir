// RUN: heir-opt --polynomial-populate-primitive-root %s | FileCheck %s

#cycl = #polynomial.int_polynomial<1 + x**256>
#ring = #polynomial.ring<coefficientType=i32, coefficientModulus=65537 : i32, polynomialModulus=#cycl>
!poly_ty = !polynomial.polynomial<ring=#ring>
!tensor_ty = tensor<256xi32, #ring>

// CHECK: func.func @poly_ntt(%[[poly0:.*]]: [[POLY_TY:.*]]) -> [[POLY_TY]] {
// CHECK:      %[[NTT_POLY0:.*]] = polynomial.ntt %[[poly0]] {[[root:.*]]} : [[POLY_TY]] -> [[INPUT_TENSOR_TYPE:.*]]
// CHECK:      %[[RES:.*]] = polynomial.intt %[[NTT_POLY0]] {[[root]]} : [[INPUT_TENSOR_TYPE]] -> {{.*}}
// CHECK:      return %[[RES]] : [[POLY_TY]]
func.func @poly_ntt(%poly0: !poly_ty) -> !poly_ty {
  %0 = polynomial.ntt %poly0 : !poly_ty -> !tensor_ty
  %1 = polynomial.intt %0 : !tensor_ty -> !poly_ty
  return %1 : !poly_ty
}

// not included in StaticRoot
#small_cycl = #polynomial.int_polynomial<1 + x**8>
#small_ring = #polynomial.ring<coefficientType=i32, coefficientModulus=65537 : i32, polynomialModulus=#small_cycl>
!small_poly_ty = !polynomial.polynomial<ring=#small_ring>
!small_tensor_ty = tensor<8xi32, #small_ring>

// CHECK: func.func @small_poly_ntt(%[[poly0:.*]]: [[POLY_TY:.*]]) -> [[POLY_TY]] {
// CHECK:      %[[NTT_POLY0:.*]] = polynomial.ntt %[[poly0]] : [[POLY_TY]] -> [[INPUT_TENSOR_TYPE:.*]]
// CHECK:      %[[RES:.*]] = polynomial.intt %[[NTT_POLY0]] : [[INPUT_TENSOR_TYPE]] -> {{.*}}
// CHECK:      return %[[RES]] : [[POLY_TY]]
func.func @small_poly_ntt(%poly0: !small_poly_ty) -> !small_poly_ty {
  %0 = polynomial.ntt %poly0 : !small_poly_ty -> !small_tensor_ty
  %1 = polynomial.intt %0 : !small_tensor_ty -> !small_poly_ty
  return %1 : !small_poly_ty
}
