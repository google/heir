// RUN: heir-opt --forward-store-to-load %s | FileCheck %s

// CHECK-LABEL: func.func @single_store
// CHECK-SAME: %[[ARG0:.*]]: memref<10xi8>
func.func @single_store(%arg0: memref<10xi8>) -> i8 {
  // CHECK-NEXT: %[[VAL:.*]] = arith.constant 7
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.store %[[VAL]], %[[ARG0]][%[[C0]]] : memref<10xi8>
  // CHECK-NEXT: return %[[VAL]] : i8
  %0 = arith.constant 7 : i8
  %c0 = arith.constant 0 : index
  memref.store %0, %arg0[%c0] : memref<10xi8>
  %1 = memref.load %arg0[%c0] : memref<10xi8>
  return %1: i8
}
