// RUN: heir-opt %s --lower-polynomial-eval=method=ps | FileCheck %s

!eval_poly_ty = !polynomial.polynomial<ring=<coefficientType=f64>>
#eval_poly_for_paterson = #polynomial.typed_float_polynomial<1.0x + 1.0x**2 + 1.0x**3 + 1.0x**4 + 1.0x**6> : !eval_poly_ty
#eval_poly_for_paterson_25_power = #polynomial.typed_float_polynomial<1.0x**26> : !eval_poly_ty

// CHECK: @test_eval_for_paterson
func.func @test_eval_for_paterson() -> f64 {
    // CHECK-NOT: polynomial.eval
    %c6 = arith.constant 6.0 : f64
    %0 = polynomial.eval #eval_poly_for_paterson, %c6 : f64
    // CHECK: %[[C6:.*]] = arith.constant 6.000000e+00 : f64
    // CHECK: %[[C1_1:.*]] = arith.constant 1.000000e+00 : f64
    // CHECK: %[[X2:.*]] = arith.mulf %[[C6]], %[[C6]] : f64
    // CHECK: %[[X3:.*]] = arith.mulf %[[C6]], %[[X2]] : f64
    // CHECK: %[[X6_PART:.*]] = arith.mulf %[[C1_1]], %[[X3]] : f64
    // CHECK: %[[C1_2:.*]] = arith.constant 1.000000e+00 : f64
    // CHECK: %[[C1_3:.*]] = arith.constant 1.000000e+00 : f64
    // CHECK: %[[X:.*]] = arith.mulf %[[C1_3]], %[[C6]] : f64
    // CHECK: %[[CONST_X:.*]] = arith.addf %[[C1_2]], %[[X]] : f64
    // CHECK: %[[X6_CONST_X:.*]] = arith.addf %[[X6_PART]], %[[CONST_X]] : f64
    // CHECK: %[[X6_MUL:.*]] = arith.mulf %[[X6_CONST_X]], %[[X3]] : f64
    // CHECK: %[[C1_4:.*]] = arith.constant 1.000000e+00 : f64
    // CHECK: %[[X_COEFF:.*]] = arith.mulf %[[C1_4]], %[[C6]] : f64
    // CHECK: %[[C1_5:.*]] = arith.constant 1.000000e+00 : f64
    // CHECK: %[[X2_COEFF:.*]] = arith.mulf %[[C1_5]], %[[X2]] : f64
    // CHECK: %[[X_X2:.*]] = arith.addf %[[X_COEFF]], %[[X2_COEFF]] : f64
    // CHECK: %[[RESULT:.*]] = arith.addf %[[X6_MUL]], %[[X_X2]] : f64
    // CHECK: return %[[RESULT]] : f64
    return %0 : f64
}

// CHECK: @test_tensor_typed_input
func.func @test_tensor_typed_input() -> tensor<8xf64> {
    // CHECK-NOT: polynomial.eval
    %c6 = arith.constant dense<6.0> : tensor<8xf64>
    %0 = polynomial.eval #eval_poly_for_paterson, %c6 : tensor<8xf64>
    // CHECK: %[[C6:.*]] = arith.constant dense<6.000000e+00> : tensor<8xf64>
    // CHECK: %[[C1_1:.*]] = arith.constant dense<1.000000e+00> : tensor<8xf64>
    // CHECK: %[[X2:.*]] = arith.mulf %[[C6]], %[[C6]] : tensor<8xf64>
    // CHECK: %[[X3:.*]] = arith.mulf %[[C6]], %[[X2]] : tensor<8xf64>
    // CHECK: %[[X6_PART:.*]] = arith.mulf %[[C1_1]], %[[X3]] : tensor<8xf64>
    // CHECK: %[[C1_2:.*]] = arith.constant dense<1.000000e+00> : tensor<8xf64>
    // CHECK: %[[C1_3:.*]] = arith.constant dense<1.000000e+00> : tensor<8xf64>
    // CHECK: %[[X:.*]] = arith.mulf %[[C1_3]], %[[C6]] : tensor<8xf64>
    // CHECK: %[[CONST_X:.*]] = arith.addf %[[C1_2]], %[[X]] : tensor<8xf64>
    // CHECK: %[[X6_CONST_X:.*]] = arith.addf %[[X6_PART]], %[[CONST_X]] : tensor<8xf64>
    // CHECK: %[[X6_MUL:.*]] = arith.mulf %[[X6_CONST_X]], %[[X3]] : tensor<8xf64>
    // CHECK: %[[C1_4:.*]] = arith.constant dense<1.000000e+00> : tensor<8xf64>
    // CHECK: %[[X_COEFF:.*]] = arith.mulf %[[C1_4]], %[[C6]] : tensor<8xf64>
    // CHECK: %[[C1_5:.*]] = arith.constant dense<1.000000e+00> : tensor<8xf64>
    // CHECK: %[[X2_COEFF:.*]] = arith.mulf %[[C1_5]], %[[X2]] : tensor<8xf64>
    // CHECK: %[[X_X2:.*]] = arith.addf %[[X_COEFF]], %[[X2_COEFF]] : tensor<8xf64>
    // CHECK: %[[RESULT:.*]] = arith.addf %[[X6_MUL]], %[[X_X2]] : tensor<8xf64>
    return %0 : tensor<8xf64>
}

// CHECK: @test_evaluating_x_powers
func.func @test_evaluating_x_powers() -> tensor<8xf64> {
    // CHECK-NOT: polynomial.eval
    %c6 = arith.constant dense<6.0> : tensor<8xf64>
    %0 = polynomial.eval #eval_poly_for_paterson_25_power, %c6 : tensor<8xf64>
    // CHECK: %[[C6:.*]] = arith.constant dense<6.000000e+00> : tensor<8xf64>
    // CHECK: %[[C1:.*]] = arith.constant dense<1.000000e+00> : tensor<8xf64>
    // CHECK: %[[X2:.*]] = arith.mulf %[[C6]], %[[C6]] : tensor<8xf64>
    // CHECK: %[[X2_COEFF:.*]] = arith.mulf %[[C1]], %[[X2]] : tensor<8xf64>
    // CHECK: %[[X3:.*]] = arith.mulf %[[C6]], %[[X2]] : tensor<8xf64>
    // CHECK: %[[X6:.*]] = arith.mulf %[[X3]], %[[X3]] : tensor<8xf64>
    // CHECK: %[[X26:.*]] = arith.mulf %[[X2_COEFF]], %[[X6]] : tensor<8xf64>
    return %0 : tensor<8xf64>
}
