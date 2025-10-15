// RUN: heir-opt --relu-canonicalizations --canonicalize --split-input-file %s | FileCheck %s

// CHECK: func.func @relu_signed
// CHECK-SAME: (%[[ARG0:.*]]: i32) -> i32
// CHECK: %[[A:.*]] = arith.constant 0
// CHECK-NEXT: arith.maxsi %[[ARG0]], %[[A]]
// CHECK: return
func.func @relu_signed(%arg0: i32) -> i32 {
  %cst_0 = arith.constant 0 : i32
  %0 = arith.cmpi sgt, %arg0, %cst_0 : i32
  %1 = arith.select %0, %arg0, %cst_0 : i32
  return %1 : i32
}

// -----

// CHECK: func.func @relu_unsigned
// CHECK-SAME: (%[[ARG0:.*]]: i32) -> i32
// CHECK: %[[A:.*]] = arith.constant 1
// CHECK-NEXT: arith.maxui %[[ARG0]], %[[A]]
// CHECK: return
func.func @relu_unsigned(%arg0: i32) -> i32 {
  %cst_1 = arith.constant 1 : i32
  %0 = arith.cmpi ugt, %arg0, %cst_1 : i32
  %1 = arith.select %0, %arg0, %cst_1 : i32
  return %1 : i32
}
