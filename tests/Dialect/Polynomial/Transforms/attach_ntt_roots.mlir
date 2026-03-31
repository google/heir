// RUN: heir-opt --attach-ntt-roots %s | FileCheck %s

// CHECK-DAG: ![[COEFF:.*]] = !mod_arith.int<7681 : i32>
// CHECK-DAG: #[[RING:.*]] = #polynomial.ring<coefficientType = ![[COEFF]], polynomialModulus = <1 + x**4>>
// CHECK-DAG: ![[POLY:.*]] = !polynomial.polynomial<ring = #[[RING]]>
// CHECK-DAG: ![[NTT_POLY:.*]] = !polynomial.polynomial<ring = #[[RING]], form = eval>

#cycl = #polynomial.int_polynomial<1 + x**4>
!coeff_ty = !mod_arith.int<7681:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#cycl>
!poly_ty = !polynomial.polynomial<ring=#ring>
!ntt_poly_ty = !polynomial.polynomial<ring=#ring, form=eval>

// CHECK: func.func @attach_roots([[ARG:%.+]]: ![[POLY]]) -> ![[POLY]] {
func.func @attach_roots(%arg0: !poly_ty) -> !poly_ty {
  // CHECK: [[NTT:%.+]] = polynomial.ntt [[ARG]] {root = #polynomial.primitive_root<value = #mod_arith.value<1925 : ![[COEFF]]> : ![[COEFF]], degree = 8 : i64>} : ![[POLY]]
  %0 = polynomial.ntt %arg0 : !poly_ty
  // CHECK: [[INTT:%.+]] = polynomial.intt [[NTT]] {root = #polynomial.primitive_root<value = #mod_arith.value<1213 : ![[COEFF]]> : ![[COEFF]], degree = 8 : i64>} : ![[NTT_POLY]]
  %1 = polynomial.intt %0 : !ntt_poly_ty
  // CHECK: return [[INTT]] : ![[POLY]]
  return %1 : !poly_ty
}
