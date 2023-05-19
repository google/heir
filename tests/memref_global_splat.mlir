// RUN: heir-opt --memref-global-replace %s | FileCheck %s

// This verifies that the memref.global was removed and that its constant values
// are forwarded to referencing affine loads when using splats.

// CHECK-LABEL: module
module {
  // CHECK-NOT: memref.global
  memref.global "private" constant @__constant_10x8x1x8xi8 : memref<10x8x1x8xi8> = dense<2>
  // CHECK-LABEL: func @main
  func.func @main() -> i8 {
    // CHECK-NOT: memref.get_global
    %0 = memref.get_global @__constant_10x8x1x8xi8 : memref<10x8x1x8xi8>
    %c_0 = arith.constant 0 : index
    %c_1 = arith.constant 1 : index
    %c_2 = arith.constant 2 : index
    // CHECK: arith.constant 2
    %10 = affine.load %0[%c_0 + %c_1, %c_2, %c_0, %c_0] : memref<10x8x1x8xi8>
    // CHECK: return
    return %10 : i8
  }
}
