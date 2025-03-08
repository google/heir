// RUN: heir-opt %s | FileCheck %s

// CHECK: !Z17_i32 = !mod_arith.int<17 : i32>
!Zp1 = !mod_arith.int<17 : i32>
// CHECK: !Z65536_i32 = !mod_arith.int<65536 : i32>
!Zp2 = !mod_arith.int<65536 : i32>

#ideal1 = #polynomial.int_polynomial<x**1024 + 1>
#ideal2 = #polynomial.int_polynomial<x**2048 + 1>

// CHECK: #ring_Z17_i32_1_x1024 = #polynomial.ring<coefficientType = !Z17_i32, polynomialModulus = <1 + x**1024>>
#ring1 = #polynomial.ring<coefficientType=!Zp1, polynomialModulus=#ideal1>
// CHECK: #ring_Z65536_i32_1_x2048 = #polynomial.ring<coefficientType = !Z65536_i32, polynomialModulus = <1 + x**2048>>
#ring2 = #polynomial.ring<coefficientType=!Zp2, polynomialModulus=#ideal2>

// CHECK: !poly = !polynomial.polynomial<ring = #ring_Z17_i32_1_x1024>
!poly = !polynomial.polynomial<ring = #ring1>
// CHECK: !poly1 = !polynomial.polynomial<ring = #ring_Z65536_i32_1_x2048>
!poly1 = !polynomial.polynomial<ring = #ring2>

// CHECK-LABEL: @test_alias
func.func @test_alias(%0 : !poly, %1 : !poly1) -> !poly {
    return %0 : !poly
}
