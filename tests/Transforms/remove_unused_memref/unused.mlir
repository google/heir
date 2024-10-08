// RUN: heir-opt --remove-unused-memref %s | FileCheck %s

// CHECK-LABEL: func.func @preserve_block_arg
// CHECK-SAME: %[[ARG0:.*]]: memref<10xi8>
func.func @preserve_block_arg(%arg0: memref<10xi8>) -> i8 {
  // CHECK-NEXT: %[[VAL:.*]] = arith.constant 7
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.store %[[VAL]], %[[ARG0]][%[[C0]]] : memref<10xi8>
  // CHECK-NEXT: return %[[VAL]] : i8
  %0 = arith.constant 7 : i8
  %c0 = arith.constant 0 : index
  memref.store %0, %arg0[%c0] : memref<10xi8>
  return %0: i8
}

// CHECK-LABEL: func.func @unused_memref
// CHECK-SAME: %[[ARG1:.*]]: i8
func.func @unused_memref(%arg0: i8) -> i8 {
  %c0 = arith.constant 0 : index
  // CHECK-NOT: memref.alloc
  %0 = memref.alloc() : memref<1xi8>
  memref.store %arg0, %0[%c0] : memref<1xi8>
  // CHECK: return %[[ARG1]] : i8
  return %arg0: i8
}

// CHECK-LABEL: func.func @unused_memref_affine_write
// CHECK-SAME: %[[ARG1:.*]]: i8
func.func @unused_memref_affine_write(%arg0: i8) -> i8 {
  %c0 = arith.constant 0 : index
  // CHECK-NOT: memref.alloc
  %0 = memref.alloc() : memref<1xi8>
  affine.store %arg0, %0[%c0] : memref<1xi8>
  // CHECK: return %[[ARG1]] : i8
  return %arg0: i8
}

// CHECK-LABEL: func.func @unused_memref_dealloc
// CHECK-SAME: %[[ARG1:.*]]: i8
func.func @unused_memref_dealloc(%arg0: i8) -> i8 {
  %c0 = arith.constant 0 : index
  // CHECK-NOT: memref.alloc
  %0 = memref.alloc() : memref<1xi8>
  affine.store %arg0, %0[%c0] : memref<1xi8>
  memref.dealloc %0 : memref<1xi8>
  // CHECK: return %[[ARG1]] : i8
  return %arg0: i8
}

// CHECK-LABEL: func.func @memref_used
// CHECK-SAME: %[[ARG1:.*]]: i8
func.func @memref_used(%arg0: i8) -> i8 {
  %c0 = arith.constant 0 : index
  // CHECK: memref.alloc
  %0 = memref.alloc() : memref<1xi8>
  affine.store %arg0, %0[%c0] : memref<1xi8>
  %1 = affine.load %0[%c0] : memref<1xi8>
  memref.dealloc %0 : memref<1xi8>
  // CHECK: return
  return %1: i8
}

// CHECK-LABEL: func.func @memref_used
// CHECK-SAME: %[[ARG1:.*]]: i8
func.func @memref_used_memref(%arg0: i8) -> i8 {
  %c0 = arith.constant 0 : index
  // CHECK: memref.alloc
  %0 = memref.alloc() : memref<1xi8>
  affine.store %arg0, %0[%c0] : memref<1xi8>
  %1 = memref.load %0[%c0] : memref<1xi8>
  memref.dealloc %0 : memref<1xi8>
  // CHECK: return
  return %1: i8
}
