// RUN: heir-opt --memref-global-replace %s 2>&1 | FileCheck %s

// This verifies that the memref.global cannot be removed when its accessor uses
// indices that cannot be resolved (i.e. the input variable).

// CHECK: MemrefGlobalLoweringPattern requires constant memref accessors
module {
  memref.global "private" constant @__constant_10x8x1x8xi8 : memref<10x8x1x8xi8> = dense<2>
  func.func @main(%arg0: i2) -> i8 {
    %0 = memref.get_global @__constant_10x8x1x8xi8 : memref<10x8x1x8xi8>
    %1 = arith.index_cast %arg0 : i2 to index
    %c_0 = arith.constant 0 : index
    %2 = affine.load %0[%c_0, %c_0, %c_0, %1] : memref<10x8x1x8xi8>
    return %2 : i8
  }
}
