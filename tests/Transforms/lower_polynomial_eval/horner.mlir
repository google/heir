// RUN: heir-opt %s --lower-polynomial-eval=method=horner | FileCheck %s

#ring_f64_ = #polynomial.ring<coefficientType = f64>
!polyty = !polynomial.polynomial<ring = #ring_f64_>
#simple_poly = #polynomial.typed_float_polynomial<4.0 + 1.0x + 1.0x**2> : !polyty
#skipped_monomials = #polynomial<typed_float_polynomial <1.0 + 4.0x**3> : !polyty>

// CHECK: @test_simple_poly
func.func @test_simple_poly() -> f64 {
    // CHECK-NOT: polynomial.eval
    %c6 = arith.constant 6.0 : f64
    %0 = polynomial.eval #simple_poly, %c6 : f64
    // CHECK: %[[C6_H:.*]] = arith.constant 6.000000e+00 : f64
    // CHECK: %[[C1_H1:.*]] = arith.constant 1.0
    // CHECK: %[[X_H:.*]] = arith.mulf %[[C1_H1]], %[[C6_H]]
    // CHECK: %[[C1_H2:.*]] = arith.constant 1.0
    // CHECK: %[[X1_H:.*]] = arith.addf %[[X_H]], %[[C1_H2]]
    // CHECK: %[[X2_H:.*]] = arith.mulf %[[X1_H]], %[[C6_H]]
    // CHECK: %[[C4_H:.*]] = arith.constant 4.0
    // CHECK: %[[RESULT:.*]] = arith.addf %[[X2_H]], %[[C4_H]]
    // CHECK: return %[[RESULT]]
    return %0 : f64
}

// CHECK: test_skipped_monomials
// CHECK-SAME: ([[arg0:%[^:]*]]: f64
func.func @test_skipped_monomials(%arg0: f64) -> f64 {
  // CHECK-NOT: polynomial.eval
  // CHECK: [[cst:%[^ ]*]] = arith.constant 4.0
  // CHECK: [[four_x:%[^ ]*]] = arith.mulf [[cst]], [[arg0]]
  // CHECK: [[four_xsq:%[^ ]*]] = arith.mulf [[four_x]], [[arg0]]
  // CHECK: [[four_xcb:%[^ ]*]] = arith.mulf [[four_xsq]], [[arg0]]
  // CHECK: [[c1:%[^ ]*]] = arith.constant 1.0
  // CHECK: [[result:%[^ ]*]] = arith.addf [[four_xcb]], [[c1]] : f64
  // CHECK: return [[result]]
  %0 = polynomial.eval #skipped_monomials, %arg0 : f64
  return %0 : f64
}

// CHECK: test_tensor_typed_input
// CHECK-SAME: ([[arg0:%[^:]*]]: tensor<8xf64>
func.func @test_tensor_typed_input(%arg0: tensor<8xf64>) -> tensor<8xf64> {
  // CHECK-NOT: polynomial.eval
  // CHECK: [[cst:%[^ ]*]] = arith.constant dense<4.0
  // CHECK: [[four_x:%[^ ]*]] = arith.mulf [[cst]], [[arg0]] : tensor<8xf64>
  // CHECK: [[four_xsq:%[^ ]*]] = arith.mulf [[four_x]], [[arg0]]
  // CHECK: [[four_xcb:%[^ ]*]] = arith.mulf [[four_xsq]], [[arg0]]
  // CHECK: [[c1:%[^ ]*]] = arith.constant dense<1.0
  // CHECK: [[result:%[^ ]*]] = arith.addf [[four_xcb]], [[c1]]
  // CHECK: return [[result]]
  %0 = polynomial.eval #skipped_monomials, %arg0 : tensor<8xf64>
  return %0 : tensor<8xf64>
}
