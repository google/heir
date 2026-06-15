// RUN: heir-opt --polynomial-to-mod-arith --mlir-print-local-scope %s | FileCheck %s

#poly = #polynomial.int_polynomial<1 + x**4>
#ring_i32 = #polynomial.ring<coefficientType=i32, polynomialModulus=#poly>
!poly_i32 = !polynomial.polynomial<ring=#ring_i32, form=eval>

// CHECK: func @eval_constant_monomial_int
// CHECK-SAME: (%[[ARG0:.*]]: i32)
func.func @eval_constant_monomial_int(%coeff: i32) -> !poly_i32 {
  %c0 = arith.constant 0 : index
  // CHECK: %[[SPLAT:.*]] = tensor.splat %[[ARG0]] : tensor<4xi32>
  // CHECK: return %[[SPLAT]]
  %0 = polynomial.monomial %coeff, %c0 : (i32, index) -> !poly_i32
  return %0 : !poly_i32
}

!coeff_ty = !mod_arith.int<17 : i32>
#ring_mod = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#poly>
!poly_mod = !polynomial.polynomial<ring=#ring_mod, form=eval>

// CHECK: func @eval_constant_monomial_mod
// CHECK-SAME: (%[[ARG0:.*]]: !mod_arith.int<17 : i32>)
func.func @eval_constant_monomial_mod(%coeff: !coeff_ty) -> !poly_mod {
  %c0 = arith.constant 0 : index
  // CHECK: %[[EXTRACTED:.*]] = mod_arith.extract %[[ARG0]] : !mod_arith.int<17 : i32> -> i32
  // CHECK: %[[SPLAT:.*]] = tensor.splat %[[EXTRACTED]] : tensor<4xi32>
  // CHECK: %[[ENCAPSULATED:.*]] = mod_arith.encapsulate %[[SPLAT]] : tensor<4xi32> -> tensor<4x!mod_arith.int<17 : i32>>
  // CHECK: return %[[ENCAPSULATED]]
  %0 = polynomial.monomial %coeff, %c0 : (!coeff_ty, index) -> !poly_mod
  return %0 : !poly_mod
}
