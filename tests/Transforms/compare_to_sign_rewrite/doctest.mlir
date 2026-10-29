// RUN: heir-opt --compare-to-sign-rewrite %s | FileCheck %s

// CHECK: func @cmpi_sgt
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32
// CHECK-DAG: %[[F0:.*]] = arith.sitofp %[[ARG0]] : i32 to f32
// CHECK-DAG: %[[F1:.*]] = arith.sitofp %[[ARG1]] : i32 to f32
// CHECK: %[[SUB:.*]] = arith.subf %[[F0]], %[[F1]] : f32
// CHECK: %[[SIGN:.*]] = math_ext.sign %[[SUB]] : f32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1.0{{.*}} : f32
// CHECK-DAG: %[[HALF:.*]] = arith.constant 5.0{{.*}} : f32
// CHECK: %[[ADD:.*]] = arith.addf %[[SIGN]], %[[ONE]] : f32
// CHECK: %[[MUL:.*]] = arith.mulf %[[ADD]], %[[HALF]] : f32
// CHECK: %[[RES:.*]] = arith.fptosi %[[MUL]] : f32 to i1
// CHECK: return %[[RES]] : i1
func.func @cmpi_sgt(%arg0: i32, %arg1: i32) -> i1 {
  %0 = arith.cmpi sgt, %arg0, %arg1 : i32
  return %0 : i1
}
