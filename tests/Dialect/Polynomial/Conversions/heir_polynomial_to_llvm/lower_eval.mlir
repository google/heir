// RUN: heir-opt --polynomial-to-mod-arith %s | FileCheck %s

!eval_poly_ty = !polynomial.polynomial<ring=<coefficientType=i64>>
#eval_poly_for_horner = #polynomial.typed_int_polynomial<4+x+x**2> : !eval_poly_ty
#eval_poly_for_paterson = #polynomial.typed_int_polynomial<x+x**2+x**3+x**4+x**6> : !eval_poly_ty

// CHECK-LABEL: @test_eval_for_horner
func.func @test_eval_for_horner() -> i64 {
    // CHECK-NOT: polynomial.eval
    %c6 = arith.constant 6 : i64
    %0 = polynomial.eval #eval_poly_for_horner, %c6 : i64
    // CHECK: %[[C6_H:.*]] = arith.constant 6 : i64
    // CHECK: %[[C1_H1:.*]] = arith.constant 1 : i64
    // CHECK: %[[X_H:.*]] = arith.muli %[[C1_H1]], %[[C6_H]] : i64
    // CHECK: %[[C1_H2:.*]] = arith.constant 1 : i64
    // CHECK: %[[X1_H:.*]] = arith.addi %[[X_H]], %[[C1_H2]] : i64
    // CHECK: %[[X2_H:.*]] = arith.muli %[[X1_H]], %[[C6_H]] : i64
    // CHECK: %[[C4_H:.*]] = arith.constant 4 : i64
    // CHECK: %[[RESULT:.*]] = arith.addi %[[X2_H]], %[[C4_H]] : i64
    // CHECK: return %[[RESULT]] : i64
    return %0 : i64
}

// CHECK-LABEL: @test_eval_for_paterson
func.func @test_eval_for_paterson() -> i64 {
    // CHECK-NOT: polynomial.eval
    %c6 = arith.constant 6 : i64
    %0 = polynomial.eval #eval_poly_for_paterson, %c6 : i64
    // CHECK: %[[C6_P:.*]] = arith.constant 6 : i64
    // CHECK: %[[C1_P1:.*]] = arith.constant 1 : i64
    // CHECK: %[[XPOW2:.*]] = arith.muli %[[C6_P]], %[[C6_P]] : i64
    // CHECK: %[[XPOW3:.*]] = arith.muli %[[XPOW2]], %[[C6_P]] : i64
    // CHECK: %[[C1_P2:.*]] = arith.constant 1 : i64
    // CHECK: %[[X_P:.*]] = arith.muli %[[C1_P2]], %[[C6_P]] : i64
    // CHECK: %[[C1_P3:.*]] = arith.constant 1 : i64
    // CHECK: %[[X2_P:.*]] = arith.muli %[[C1_P3]], %[[XPOW2]] : i64
    // CHECK: %[[X12_P:.*]] = arith.addi %[[X_P]], %[[X2_P]] : i64
    // CHECK: %[[C1_P4:.*]] = arith.constant 1 : i64
    // CHECK: %[[C1_P5:.*]] = arith.constant 1 : i64
    // CHECK: %[[XX_P:.*]] = arith.muli %[[C1_P5]], %[[C6_P]] : i64
    // CHECK: %[[XX1_P:.*]] = arith.addi %[[C1_P4]], %[[XX_P]] : i64
    // CHECK: %[[C1_P6:.*]] = arith.constant 1 : i64
    // CHECK: %[[X3_P:.*]] = arith.muli %[[C1_P6]], %[[XPOW3]] : i64
    // CHECK: %[[X13_P:.*]] = arith.addi %[[X3_P]], %[[XX1_P]] : i64
    // CHECK: %[[X6_P:.*]] = arith.muli %[[X13_P]], %[[XPOW3]] : i64
    // CHECK: %[[RESULT:.*]] = arith.addi %[[X6_P]], %[[X12_P]] : i64
    // CHECK: return %[[RESULT]] : i64
    return %0 : i64
}
