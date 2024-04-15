// RUN: heir-opt --polynomial-to-standard %s | FileCheck %s

!poly_ty = !_polynomial.polynomial<<cmod=18446744073709551616, ideal=#_polynomial.polynomial<-1 + x**2047>>>

// CHECK-LABEL: func.func @test_lower_mul_scalar
// CHECK-SAME: (%[[ARG:.*]]: [[T:.*]])
func.func @test_lower_mul_scalar(%arg0: !poly_ty) -> !poly_ty {
  // CHECK: %[[C2:.*]] = arith.constant 2 : i64
  %c2 = arith.constant 2 : i64
  // CHECK: %[[SPLAT:.*]] = tensor.splat %[[C2]] : [[T]]
  // CHECK: arith.muli %[[ARG]], %[[SPLAT]] : [[T]]
  %8 = _polynomial.mul_scalar %arg0, %c2 : !poly_ty, i64
  return %8 : !poly_ty
}

// CHECK-LABEL: func.func @test_lower_mul_scalar_lift_i32
// CHECK-SAME: (%[[ARG:.*]]: [[T:.*]])
func.func @test_lower_mul_scalar_lift_i32(%arg0: !poly_ty) -> !poly_ty {
  // CHECK: %[[C2:.*]] = arith.constant 2 : i32
  %c2 = arith.constant 2 : i32
  // CHECK: %[[C2_EXT:.*]] = arith.extsi %[[C2]] : i32 to i64
  // CHECK: %[[SPLAT:.*]] = tensor.splat %[[C2_EXT]] : [[T]]
  // CHECK: arith.muli %[[ARG]], %[[SPLAT]] : [[T]]
  %8 = _polynomial.mul_scalar %arg0, %c2 : !poly_ty, i32
  return %8 : !poly_ty
}
