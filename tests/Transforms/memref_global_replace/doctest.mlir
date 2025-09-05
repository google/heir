// RUN: heir-opt --memref-global-replace %s | FileCheck %s

// CHECK: func.func @main() -> i16
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %c2 = arith.constant 2 : index
// CHECK-NEXT:   %c8_i16 = arith.constant 8 : i16
// CHECK-NEXT:   return %c8_i16 : i16
// CHECK-NOT: memref.global

module {
  memref.global "private" constant @__constant_8xi16 : memref<2x4xi16> = dense<[[-10, 20, 3, 4], [5, 6, 7, 8]]>
  func.func @main() -> i16 {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = memref.get_global @__constant_8xi16 : memref<2x4xi16>
    %1 = affine.load %0[%c1, %c1 + %c2] : memref<2x4xi16>
    return %1 : i16
  }
}
