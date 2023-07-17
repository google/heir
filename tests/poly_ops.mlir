// RUN: heir-opt %s | FileCheck %s

// This simply tests for syntax.

module {
// CHECK-LABEL: func @fooFunc
  func.func @fooFunc(%arg0: !poly.poly, %arg1: !poly.poly) -> !poly.poly {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1_i32 = arith.constant 1 : i32
    // CHECK: poly.mul
    %0 = poly.mul(%arg0, %arg1) : !poly.poly
    // CHECK: poly.extract_slice
    %1 = poly.extract_slice(%0, %c0, %c3) : (!poly.poly, index, index) -> tensor<3xi32>
    // CHECK: poly.from_coeffs
    %2 = poly.from_coeffs(%1) : (tensor<3xi32>) -> !poly.poly
    // CHECK: poly.add
    %3 = poly.add(%arg0, %arg1, %2) : !poly.poly
    // CHECK: poly.get_coeff
    %4 = poly.get_coeff(%3, %c0) : (!poly.poly, index) -> i32
    // CHECK: poly.mul_constant
    %5 = poly.mul_constant(%3, %4) : (!poly.poly, i32) -> !poly.poly
    return %5 : !poly.poly
  }
}
