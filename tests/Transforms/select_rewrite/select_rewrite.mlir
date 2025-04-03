// RUN: heir-opt --select-rewrite %s | FileCheck %s

// CHECK: func @scalar_arith_select
// CHECK-SAME: [[COND:%.*]]: i1, [[LHS:%.*]]: i32, [[RHS:%.*]]: i32
func.func @scalar_arith_select(%cond : i1, %lhs : i32, %rhs : i32) ->  i32 {
    // CHECK: [[ONE:%.*]] = arith.constant true
    // CHECK-DAG: [[NCOND:%.*]] = arith.subi [[ONE]], [[COND]]
    // CHECK-DAG: [[XCOND:%.*]] = arith.extui [[COND]]
    // CHECK-DAG: [[XNCOND:%.*]] = arith.extui [[NCOND]]
    // CHECK-DAG: [[MLHS:%.*]] = arith.muli [[XCOND]], [[LHS]]
    // CHECK-DAG: [[MRHS:%.*]] = arith.muli [[XNCOND]], [[RHS]]
    // CHECK: [[RES:%.*]] = arith.addi [[MLHS]], [[MRHS]]
    // CHECK-NOT: arith.select
    %0 = arith.select %cond, %lhs, %rhs : i32
    // CHECK: return [[RES:%.*]] : i32
    return %0 : i32
}

// CHECK: func @vector_arith_select
// CHECK: [[COND:%.*]]: tensor<2xi1>, [[LHS:%.*]]: tensor<2xi32>, [[RHS:%.*]]: tensor<2xi32>
func.func @vector_arith_select(%cond : tensor<2xi1>, %lhs : tensor<2xi32>, %rhs : tensor<2xi32>) ->  tensor<2xi32> {
    // CHECK: [[ONE:%.*]] = arith.constant dense<true>
    // CHECK-DAG: [[NCOND:%.*]] = arith.subi [[ONE]], [[COND]]
    // CHECK-DAG: [[XCOND:%.*]] = arith.extui [[COND]]
    // CHECK-DAG: [[XNCOND:%.*]] = arith.extui [[NCOND]]
    // CHECK-DAG: [[MLHS:%.*]] = arith.muli [[XCOND]], [[LHS]]
    // CHECK-DAG: [[MRHS:%.*]] = arith.muli [[XNCOND]], [[RHS]]
    // CHECK: [[RES:%.*]] = arith.addi [[MLHS]], [[MRHS]]
    // CHECK-NOT: arith.select
    %0 = arith.select %cond, %lhs, %rhs :  tensor<2xi1>, tensor<2xi32>
    // CHECK: return [[RES:%.*]] : tensor<2xi32>
    return %0 :  tensor<2xi32>
}

// CHECK: func @mixed_arith_select
// CHECK: [[COND:%.*]]: i1, [[LHS:%.*]]: tensor<2xi32>, [[RHS:%.*]]: tensor<2xi32>
func.func @mixed_arith_select(%cond : i1, %lhs : tensor<2xi32>, %rhs : tensor<2xi32>) ->  tensor<2xi32> {
    // CHECK: [[ONE:%.*]] = arith.constant dense<true>
    // CHECK-DAG: [[SPLAT:%.*]] = tensor.splat [[COND]]
    // CHECK-DAG: [[NCOND:%.*]] = arith.subi [[ONE]], [[SPLAT]]
    // CHECK-DAG: [[XCOND:%.*]] = arith.extui [[SPLAT]]
    // CHECK-DAG: [[XNCOND:%.*]] = arith.extui [[NCOND]]
    // CHECK-DAG: [[MLHS:%.*]] = arith.muli [[XCOND]], [[LHS]]
    // CHECK-DAG: [[MRHS:%.*]] = arith.muli [[XNCOND]], [[RHS]]
    // CHECK: [[RES:%.*]] = arith.addi [[MLHS]], [[MRHS]]
    // CHECK-NOT: arith.select
    %0 = arith.select %cond, %lhs, %rhs : i1, tensor<2xi32>
    // CHECK: return [[RES:%.*]] : tensor<2xi32>
    return %0 : tensor<2xi32>
}

// CHECK func @float_arith_select
// CHECK: [[COND:%.*]]: i1, [[LHS:%.*]]: f32, [[RHS:%.*]]: f32
func.func @float_arith_select(%cond : i1, %lhs : f32, %rhs : f32) ->  f32 {
    // CHECK: [[ONE:%.*]] = arith.constant true
    // CHECK-DAG: [[NCOND:%.*]] = arith.subi [[ONE]], [[COND]]
    // CHECK-DAG: [[XCOND:%.*]] = arith.uitofp  [[COND]]
    // CHECK-DAG: [[XNCOND:%.*]] = arith.uitofp  [[NCOND]]
    // CHECK-DAG: [[MLHS:%.*]] = arith.mulf [[XCOND]], [[LHS]]
    // CHECK-DAG: [[MRHS:%.*]] = arith.mulf [[XNCOND]], [[RHS]]
    // CHECK: [[RES:%.*]] = arith.addf [[MLHS]], [[MRHS]]
    // CHECK-NOT: arith.select
    %0 = arith.select %cond, %lhs, %rhs : f32
    // CHECK: return [[RES:%.*]] : f32
    return %0 : f32
}

// CHECK: func @vector_float_arith_select
// CHECK: [[COND:%.*]]: tensor<2xi1>, [[LHS:%.*]]: tensor<2xf32>, [[RHS:%.*]]: tensor<2xf32>
func.func @vector_float_arith_select(%cond : tensor<2xi1>, %lhs : tensor<2xf32>, %rhs : tensor<2xf32>) ->  tensor<2xf32> {
    // CHECK: [[ONE:%.*]] = arith.constant dense<true>
    // CHECK-DAG: [[NCOND:%.*]] = arith.subi [[ONE]], [[COND]]
    // CHECK-DAG: [[XCOND:%.*]] = arith.uitofp  [[COND]]
    // CHECK-DAG: [[XNCOND:%.*]] = arith.uitofp  [[NCOND]]
    // CHECK-DAG: [[MLHS:%.*]] = arith.mulf [[XCOND]], [[LHS]]
    // CHECK-DAG: [[MRHS:%.*]] = arith.mulf [[XNCOND]], [[RHS]]
    // CHECK: [[RES:%.*]] = arith.addf [[MLHS]], [[MRHS]]
    // CHECK-NOT: arith.select
    %0 = arith.select %cond, %lhs, %rhs :  tensor<2xi1>, tensor<2xf32>
    // CHECK: return [[RES:%.*]] : tensor<2xf32>
    return %0 :  tensor<2xf32>
}
