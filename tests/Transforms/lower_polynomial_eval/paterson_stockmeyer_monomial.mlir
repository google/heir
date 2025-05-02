// RUN: heir-opt %s --lower-polynomial-eval=method=ps | FileCheck %s

!eval_poly_ty = !polynomial.polynomial<ring=<coefficientType=f64>>
#eval_poly_for_paterson = #polynomial.typed_float_polynomial<1.0x + 1.0x**2 + 1.0x**3 + 1.0x**4 + 1.0x**6> : !eval_poly_ty
#eval_poly_for_paterson_25_power = #polynomial.typed_float_polynomial<1.0x**26> : !eval_poly_ty

// CHECK: @test_eval_for_paterson
func.func @test_eval_for_paterson() -> f64 {
    // CHECK-NOT: polynomial.eval
    %c6 = arith.constant 6.0 : f64
    %0 = polynomial.eval #eval_poly_for_paterson, %c6 : f64
    // CHECK: %[[C6_P:.*]] = arith.constant 6.0
    // CHECK: %[[C1_P1:.*]] = arith.constant 1.0
    // CHECK: %[[XPOW2:.*]] = arith.mulf %[[C6_P]], %[[C6_P]] : f64
    // CHECK: %[[XPOW3:.*]] = arith.mulf %[[C6_P]], %[[XPOW2]] : f64
    // CHECK: %[[C1_P2:.*]] = arith.constant 1.0
    // CHECK: %[[X_P:.*]] = arith.mulf %[[C1_P2]], %[[C6_P]] : f64
    // CHECK: %[[C1_P3:.*]] = arith.constant 1.0
    // CHECK: %[[X2_P:.*]] = arith.mulf %[[C1_P3]], %[[XPOW2]] : f64
    // CHECK: %[[X12_P:.*]] = arith.addf %[[X_P]], %[[X2_P]] : f64
    // CHECK: %[[C1_P4:.*]] = arith.constant 1.0
    // CHECK: %[[C1_P5:.*]] = arith.constant 1.0
    // CHECK: %[[XX_P:.*]] = arith.mulf %[[C1_P5]], %[[C6_P]] : f64
    // CHECK: %[[XX1_P:.*]] = arith.addf %[[C1_P4]], %[[XX_P]] : f64
    // CHECK: %[[C1_P6:.*]] = arith.constant 1.0
    // CHECK: %[[X3_P:.*]] = arith.mulf %[[C1_P6]], %[[XPOW3]] : f64
    // CHECK: %[[X13_P:.*]] = arith.addf %[[X3_P]], %[[XX1_P]] : f64
    // CHECK: %[[X6_P:.*]] = arith.mulf %[[X13_P]], %[[XPOW3]] : f64
    // CHECK: %[[RESULT:.*]] = arith.addf %[[X6_P]], %[[X12_P]] : f64
    // CHECK: return %[[RESULT]] : f64
    return %0 : f64
}

// CHECK: @test_tensor_typed_input
func.func @test_tensor_typed_input() -> tensor<8xf64> {
    // CHECK-NOT: polynomial.eval
    %c6 = arith.constant dense<6.0> : tensor<8xf64>
    %0 = polynomial.eval #eval_poly_for_paterson, %c6 : tensor<8xf64>
    // CHECK: %[[C6_P:.*]] = arith.constant dense<6.0
    // CHECK: %[[C1_P1:.*]] = arith.constant dense<1.0
    // CHECK: %[[XPOW2:.*]] = arith.mulf %[[C6_P]], %[[C6_P]] : tensor<8xf64>
    // CHECK: %[[XPOW3:.*]] = arith.mulf %[[C6_P]], %[[XPOW2]] : tensor<8xf64>
    // CHECK: %[[C1_P2:.*]] = arith.constant dense<1.0
    // CHECK: %[[X_P:.*]] = arith.mulf %[[C1_P2]], %[[C6_P]]
    // CHECK: %[[C1_P3:.*]] = arith.constant dense<1.0
    // CHECK: %[[X2_P:.*]] = arith.mulf %[[C1_P3]], %[[XPOW2]]
    // CHECK: %[[X12_P:.*]] = arith.addf %[[X_P]], %[[X2_P]]
    // CHECK: %[[C1_P4:.*]] = arith.constant dense<1.0
    // CHECK: %[[C1_P5:.*]] = arith.constant dense<1.0
    // CHECK: %[[XX_P:.*]] = arith.mulf %[[C1_P5]], %[[C6_P]]
    // CHECK: %[[XX1_P:.*]] = arith.addf %[[C1_P4]], %[[XX_P]]
    // CHECK: %[[C1_P6:.*]] = arith.constant dense<1.0
    // CHECK: %[[X3_P:.*]] = arith.mulf %[[C1_P6]], %[[XPOW3]]
    // CHECK: %[[X13_P:.*]] = arith.addf %[[X3_P]], %[[XX1_P]]
    // CHECK: %[[X6_P:.*]] = arith.mulf %[[X13_P]], %[[XPOW3]]
    // CHECK: %[[RESULT:.*]] = arith.addf %[[X6_P]], %[[X12_P]]
    return %0 : tensor<8xf64>
}

// CHECK: @test_evaluating_x_powers
func.func @test_evaluating_x_powers() -> tensor<8xf64> {
    // CHECK-NOT: polynomial.eval
    %c6 = arith.constant dense<6.0> : tensor<8xf64>
    %0 = polynomial.eval #eval_poly_for_paterson_25_power, %c6 : tensor<8xf64>
    // CHECK: %[[C6_P:.*]] = arith.constant dense<6.0
    // CHECK: %[[XPOW2:.*]] = arith.mulf %[[C6_P]], %[[C6_P]] : tensor<8xf64>
    // CHECK: %[[XPOW3:.*]] = arith.mulf %[[C6_P]], %[[XPOW2]] : tensor<8xf64>
    // CHECK: %[[XPOW4:.*]] = arith.mulf %[[XPOW2]], %[[XPOW2]] : tensor<8xf64>
    // CHECK: %[[XPOW5:.*]] = arith.mulf %[[XPOW2]], %[[XPOW3]] : tensor<8xf64>
    // CHECK: %[[XPOW6:.*]] = arith.mulf %[[XPOW3]], %[[XPOW3]] : tensor<8xf64>
    return %0 : tensor<8xf64>
}
