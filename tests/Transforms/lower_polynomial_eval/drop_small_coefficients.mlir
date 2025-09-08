// RUN: heir-opt %s --lower-polynomial-eval="method=horner min-coefficient-threshold=1e-10" | FileCheck %s --check-prefix=CHECK-HORNER
// RUN: heir-opt %s --lower-polynomial-eval="method=ps min-coefficient-threshold=1e-10" | FileCheck %s --check-prefix=CHECK-PS

#ring_f64_ = #polynomial.ring<coefficientType = f64>
!polyty = !polynomial.polynomial<ring = #ring_f64_>

#poly_with_small_coeffs = #polynomial.typed_float_polynomial<1.0 + 1.0e-15x + 2.0x**2 + 1.0e-16x**3> : !polyty
#poly_all_small = #polynomial.typed_float_polynomial<1.0e-15 + 1.0e-16x + 1.0e-17x**2> : !polyty
#poly_mixed = #polynomial.typed_float_polynomial<1.0e-15 + 5.0x + 1.0e-14x**2 + 3.0x**3> : !polyty

// CHECK-HORNER: @test_drop_small_coefficients
func.func @test_drop_small_coefficients() -> f64 {
    // The small coefficients (1e-15 and 1e-16) should be dropped, leaving only 1.0 + 2.0x**2
    // CHECK-HORNER-NOT: polynomial.eval
    // CHECK-HORNER: %[[C6:.*]] = arith.constant 6.000000e+00 : f64
    // CHECK-HORNER: %[[C2:.*]] = arith.constant 2.000000e+00 : f64
    // CHECK-HORNER: %[[X1:.*]] = arith.mulf %[[C2]], %[[C6]]
    // CHECK-HORNER: %[[X2:.*]] = arith.mulf %[[X1]], %[[C6]]
    // CHECK-HORNER: %[[C1:.*]] = arith.constant 1.000000e+00 : f64
    // CHECK-HORNER: %[[RESULT:.*]] = arith.addf %[[X2]], %[[C1]]
    // CHECK-HORNER: return %[[RESULT]]
    %c6 = arith.constant 6.0 : f64
    %0 = polynomial.eval #poly_with_small_coeffs, %c6 : f64
    return %0 : f64
}

// CHECK-HORNER: @test_all_small_coefficients
func.func @test_all_small_coefficients() -> f64 {
    // All coefficients should be dropped, result should be zero
    // CHECK-HORNER-NOT: polynomial.eval
    // CHECK-HORNER: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
    // CHECK-HORNER: return %[[ZERO]]
    %c6 = arith.constant 6.0 : f64
    %0 = polynomial.eval #poly_all_small, %c6 : f64
    return %0 : f64
}

// CHECK-HORNER: @test_mixed_coefficients
func.func @test_mixed_coefficients() -> f64 {
    // Only large coefficients (5.0x and 3.0x**3) should remain
    // CHECK-HORNER-NOT: polynomial.eval
    // CHECK-HORNER: %[[C6:.*]] = arith.constant 6.000000e+00 : f64
    // CHECK-HORNER: %[[C3:.*]] = arith.constant 3.000000e+00 : f64
    // CHECK-HORNER: %[[X1:.*]] = arith.mulf %[[C3]], %[[C6]]
    // CHECK-HORNER: %[[X2:.*]] = arith.mulf %[[X1]], %[[C6]]
    // CHECK-HORNER: %[[C5:.*]] = arith.constant 5.000000e+00 : f64
    // CHECK-HORNER: %[[X3:.*]] = arith.addf %[[X2]], %[[C5]]
    // CHECK-HORNER: %[[RESULT:.*]] = arith.mulf %[[X3]], %[[C6]]
    // CHECK-HORNER: return %[[RESULT]]
    %c6 = arith.constant 6.0 : f64
    %0 = polynomial.eval #poly_mixed, %c6 : f64
    return %0 : f64
}

// CHECK-PS: @test_drop_small_coefficients_ps
func.func @test_drop_small_coefficients_ps() -> f64 {
    // Similar test but using Paterson-Stockmeyer method
    // CHECK-PS-NOT: polynomial.eval
    %c6 = arith.constant 6.0 : f64
    %0 = polynomial.eval #poly_with_small_coeffs, %c6 : f64
    return %0 : f64
}

// CHECK-PS: @test_all_small_coefficients_ps
func.func @test_all_small_coefficients_ps() -> f64 {
    // All coefficients should be dropped, result should be zero
    // CHECK-PS-NOT: polynomial.eval
    // CHECK-PS: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
    // CHECK-PS: return %[[ZERO]]
    %c6 = arith.constant 6.0 : f64
    %0 = polynomial.eval #poly_all_small, %c6 : f64
    return %0 : f64
}
