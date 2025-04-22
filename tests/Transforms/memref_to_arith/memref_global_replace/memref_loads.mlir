// RUN: heir-opt --memref-global-replace %s | FileCheck %s

// CHECK: module
module {
  // CHECK-NOT: memref.global
  memref.global "private" constant @__constant_10x8x1x8xi8 : memref<10x8x1x8xi8> = dense<2>
  // CHECK: func @memref_load
  func.func @memref_load() -> i8 {
    // CHECK-NOT: memref.get_global
    // CHECK-NOT: memref.load
    %0 = memref.get_global @__constant_10x8x1x8xi8 : memref<10x8x1x8xi8>
    %c_0 = arith.constant 0 : index
    %c_1 = arith.constant 1 : index
    %c_2 = arith.constant 2 : index
    // CHECK: %[[v0:.*]] = arith.constant 2 : i8
    %10 = memref.load %0[%c_1, %c_2, %c_0, %c_0] : memref<10x8x1x8xi8>
    // CHECK: return %[[v0]]
    return %10 : i8
  }
}
