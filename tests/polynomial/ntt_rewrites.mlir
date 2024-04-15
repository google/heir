// RUN: heir-opt --convert-polynomial-mul-to-ntt %s | FileCheck %s

#ideal = #_polynomial.polynomial<1 + x**4>
#ring = #_polynomial.ring<cmod=17, ideal=#ideal, root=2>
!poly_ty = !_polynomial.polynomial<#ring>

// CHECK: func.func @rewrite_poly_mul(%[[poly0:.*]]: [[POLY_TY:.*]], %[[poly1:.*]]: [[POLY_TY]]) -> [[POLY_TY]] {
// CHECK:      %[[CMOD:.*]] = arith.constant dense<17> : [[INTERMEDIATE_TENSOR_TYPE:.*]]
// CHECK:      %[[NTT_POLY0:.*]] = _polynomial.ntt %[[poly0]] : [[POLY_TY]] -> [[INPUT_TENSOR_TYPE:.*]]
// CHECK:      %[[NTT_POLY1:.*]] = _polynomial.ntt %[[poly1]] : [[POLY_TY]] -> [[INPUT_TENSOR_TYPE]]
// CHECK:      %[[NTT_EXT0:.*]] = arith.extui %[[NTT_POLY0]] : [[INPUT_TENSOR_TYPE]] to [[INTERMEDIATE_TENSOR_TYPE]]
// CHECK:      %[[NTT_EXT1:.*]] = arith.extui %[[NTT_POLY1]] : [[INPUT_TENSOR_TYPE]] to [[INTERMEDIATE_TENSOR_TYPE]]
// CHECK:      %[[NTT_MUL:.*]] = arith.muli %[[NTT_EXT0]], %[[NTT_EXT1]] : [[INTERMEDIATE_TENSOR_TYPE]]
// CHECK:      %[[NTT_MOD:.*]] = arith.remui %[[NTT_MUL]], %[[CMOD]] : [[INTERMEDIATE_TENSOR_TYPE]]
// CHECK:      %[[NTT_RES:.*]] = arith.trunci %[[NTT_MOD]] : [[INTERMEDIATE_TENSOR_TYPE]] to [[INPUT_TENSOR_TYPE]]
// CHECK:      %[[RES:.*]] = _polynomial.intt %[[NTT_RES]] : [[INPUT_TENSOR_TYPE]] -> {{.*}}
// CHECK:      return %[[RES]] : [[POLY_TY]]
func.func @rewrite_poly_mul(%poly0: !poly_ty, %poly1: !poly_ty) -> !poly_ty {
  %poly = _polynomial.mul(%poly0, %poly1) : !poly_ty
  return %poly : !poly_ty
}

#bad_ideal = #_polynomial.polynomial<1 + x**6>
#bad_ring = #_polynomial.ring<cmod=17, ideal=#bad_ideal>
!bad_poly_ty = !_polynomial.polynomial<#bad_ring>

// CHECK: func.func @rewrite_bad_poly_mul
// CHECK:      %[[POLYMUL:.*]] = _polynomial.mul
// CHECK:      return %[[POLYMUL]]
func.func @rewrite_bad_poly_mul(%poly0: !bad_poly_ty, %poly1: !bad_poly_ty) -> !bad_poly_ty {
  %poly = _polynomial.mul(%poly0, %poly1) : !bad_poly_ty
  return %poly : !bad_poly_ty
}
