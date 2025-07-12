// RUN: heir-opt %s --mlir-to-secret-arithmetic | FileCheck %s
// TODO (#1929): Improve test with better CHECKs and add an end-to-end test with --mlir-to-ckks

// CHECK: func.func @secret_loop_index_step
func.func @secret_loop_index_step(%arg0: i32 {secret.secret}, %arg1: i32 {secret.secret}) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %arg0 : i32 to index
    // CHECK: affine.for
    // CHECK-NOT: scf.for
    %1 = scf.for %arg2 = %c0 to %0 step %c1 iter_args(%arg3 = %c1_i32) -> (i32) {
        %2 = arith.muli %arg3, %arg0 : i32
        scf.yield %2 : i32
    } {upper = 6}
    return %1 : i32
}

// When scf.for uses a signless integer step, --convert-secret-for-to-static-for would forget to issue arith.index_cast
// CHECK: func.func @secret_loop_signless_integer_step
func.func @secret_loop_signless_integer_step(%arg0: i32 {secret.secret}, %arg1: i32 {secret.secret}) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK-NOT: scf.for
    // CHECK-not: arith.cmpi
    %1 = scf.for %arg2 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg3 = %c1_i32) -> (i32) : i32 {
        %2 = arith.muli %arg3, %arg0 : i32
        scf.yield %2 : i32
    } {upper = 6}
    return %1 : i32
}

// CHECK: func.func @float_secret_loop_index_step
func.func @float_secret_loop_index_step(%arg0: f32 {secret.secret}, %arg1: f32 {secret.secret}) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_f32 = arith.constant 1.0 : f32
    %0 = arith.fptosi %arg0 : f32 to i32
    %1 = arith.index_cast %0 : i32 to index
    // CHECK-NOT: scf.for
    // CHECK-not: arith.cmpi
    %2 = scf.for %arg2 = %c0 to %1 step %c1 iter_args(%arg3 = %c1_f32) -> (f32) {
        %2 = arith.mulf %arg3, %arg0 : f32
        scf.yield %2 : f32
    } {upper = 6}
    return %2 : f32
}

// When scf.for uses a signless integer step, --convert-secret-for-to-static-for would forget to issue arith.index_cast
// CHECK: func.func @float_secret_loop_signless_integer_step
func.func @float_secret_loop_signless_integer_step(%arg0: f32 {secret.secret}, %arg1: f32 {secret.secret}) -> f32 {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_f32 = arith.constant 1.0 : f32
    %0 = arith.fptosi %arg0 : f32 to i32
    // CHECK-NOT: scf.for
    // CHECK-not: arith.cmpi
    %1 = scf.for %arg2 = %c0_i32 to %0 step %c1_i32 iter_args(%arg3 = %c1_f32) -> (f32) : i32 {
        %2 = arith.mulf %arg3, %arg0 : f32
        scf.yield %2 : f32
    } {upper = 6}
    return %1 : f32
}
