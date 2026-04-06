// RUN: heir-opt --polynomial-to-mod-arith --mlir-print-local-scope %s | FileCheck %s

#poly = #polynomial.int_polynomial<1 + x**1024>
!coeff_ty = !mod_arith.int<2837465:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#poly>
!poly_mod_arith = !polynomial.polynomial<ring=#ring, form = eval>

// CHECK: func @test_mul_eval
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024x!mod_arith.int<2837465 : i32>>, %[[ARG1:.*]]: tensor<1024x!mod_arith.int<2837465 : i32>>)
func.func @test_mul_eval(%p0 : !poly_mod_arith, %p1 : !poly_mod_arith) -> !poly_mod_arith {
  // CHECK: %[[MUL:.*]] = mod_arith.mul %[[ARG0]], %[[ARG1]] : tensor<1024x!mod_arith.int<2837465 : i32>>
  // CHECK: return %[[MUL]]
  %1 = polynomial.mul %p0, %p1 : !poly_mod_arith
  return %1 : !poly_mod_arith
}

#ring2 = #polynomial.ring<coefficientType=i32, polynomialModulus=#poly>
!poly_i32 = !polynomial.polynomial<ring=#ring2, form = eval>

// CHECK: func @test_mul_eval_int
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024xi32>, %[[ARG1:.*]]: tensor<1024xi32>)
func.func @test_mul_eval_int(%p0 : !poly_i32, %p1 : !poly_i32) -> !poly_i32 {
  // CHECK: %[[MUL:.*]] = arith.muli %[[ARG0]], %[[ARG1]] : tensor<1024xi32>
  // CHECK: return %[[MUL]]
  %1 = polynomial.mul %p0, %p1 : !poly_i32
  return %1 : !poly_i32
}

#ring3 = #polynomial.ring<coefficientType=f32, polynomialModulus=#poly>
!poly_f32 = !polynomial.polynomial<ring=#ring3, form = eval>

// CHECK: func @test_mul_eval_float
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024xf32>, %[[ARG1:.*]]: tensor<1024xf32>)
func.func @test_mul_eval_float(%p0 : !poly_f32, %p1 : !poly_f32) -> !poly_f32 {
  // CHECK: %[[MUL:.*]] = arith.mulf %[[ARG0]], %[[ARG1]] : tensor<1024xf32>
  // CHECK: return %[[MUL]]
  %1 = polynomial.mul %p0, %p1 : !poly_f32
  return %1 : !poly_f32
}
