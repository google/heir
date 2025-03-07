// RUN: heir-opt --polynomial-to-mod-arith %s | FileCheck %s

!eval_poly_ty = !polynomial.polynomial<ring=<coefficientType=i64>>
#eval_poly_for_horner = #polynomial.typed_int_polynomial<4+x+x**2> : !eval_poly_ty
#eval_poly_for_paterson = #polynomial.typed_int_polynomial<x+x**2+x**3+x**4+x**6> : !eval_poly_ty


// CHECK-LABEL : @test_eval_for_horner
func.func @test_eval_for_horner() -> i64 {
    // CHECK-NOT: polynomial.eval
    %c6 = arith.constant 6 : i64
    %0 = polynomial.eval #eval_poly_for_horner, %c6 : i64
    // CHECK: %[[C6:.*]] = arith.constant 6 : i64
    // CHECK: %[[C1:.*]] = arith.constant 1 : i64
    // CHECK: arith.muli
    // CHECK: %[[C1:.*]] = arith.constant 1 : i64
    // CHECK: arith.addi
    // CHECK: arith.muli
    // CHECK: %[[C4:.*]] = arith.constant 4 : i64
    // CHECK: arith.addi
    // CHECK: return %[[RESULT:.*]] : [[TYPE:.*]]
    return %0 : i64
}

// CHECK-LABEL : @test_eval_for_paterson
func.func @test_eval_for_paterson() -> i64 {
    // CHECK-NOT: polynomial.eval
    %c7 = arith.constant 6 : i64
    %0 = polynomial.eval #eval_poly_for_paterson, %c7 : i64
    // CHECK: %[[C6:.*]] = arith.constant 6 : i64
    // CHECK: %[[C1:.*]] = arith.constant 1 : i64
    // CHECK: arith.muli
    // CHECK: arith.muli
    // CHECK: %[[C1:.*]] = arith.constant 1 : i64
    // CHECK: arith.muli
    // CHECK: %[[C1:.*]] = arith.constant 1 : i64
    // CHECK: arith.muli
    // CHECK: arith.addi
    // CHECK: %[[C1:.*]] = arith.constant 1 : i64
    // CHECK: %[[C1:.*]] = arith.constant 1 : i64
    // CHECK: arith.muli
    // CHECK: arith.addi
    // CHECK: %[[C1:.*]] = arith.constant 1 : i64
    // CHECK: arith.muli
    // CHECK: arith.addi
    // CHECK: arith.muli
    // CHECK: arith.addi
    // CHECK: return %[[RESULT:.*]] : [[TYPE:.*]]
    return %0 : i64
}
