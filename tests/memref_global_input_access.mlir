// RUN: heir-opt --memref-global-replace %s 2>&1 | FileCheck %s

// This verifies that a memref.global cannot be removed when its accessor uses
// indices that cannot be resolved (i.e. the input variable) but that other
// accesses are replaced.

module {
  // CHECK: memref.global "private" constant  [[MEM1:@[a-z0-9_]+]]
  memref.global "private" constant @__constant_10x8x1x8xi8 : memref<10x8x1x8xi8> = dense<2>
  // CHECK: memref.global "private" constant  [[MEM2:@[a-z0-9_]+]]
  memref.global "private" constant @__constant_2x2xi8 : memref<2x2xi8> = dense<[[11, 12], [13, 14]]>
  // __constant_3xi8 can be removed because it only has accessors with constant indices.
  // CHECK-NOT: memref.global
  memref.global "private" constant @__constant_3xi8 : memref<3xi8> = dense<[22, 23, 24]>
  // CHECK: func.func @main
  func.func @main(%arg0: i2) -> i8 {
    // CHECK: [[GETMEM1:%[a-z0-9_]+]] = memref.get_global {{.*}}[[MEM1]]
    %0 = memref.get_global @__constant_10x8x1x8xi8 : memref<10x8x1x8xi8>
    // CHECK: [[GETMEM2:%[a-z0-9_]+]] = memref.get_global {{.*}}[[MEM2]]
    %1 = memref.get_global @__constant_2x2xi8 : memref<2x2xi8>
    // CHECK-NOT: memref.get_global
    %2 = memref.get_global @__constant_3xi8 : memref<3xi8>
    %3 = arith.index_cast %arg0 : i2 to index
    %c0 = arith.constant 0 : index
    // CHECK: affine.load {{.*}}[[GETMEM2]]
    %4 = affine.load %1[%c0, %3] : memref<2x2xi8>
    // CHECK: arith.constant 2
    %5 = affine.load %0[%c0, %c0, %c0, %c0] : memref<10x8x1x8xi8>
    %6 = arith.addi %4, %5 : i8
    // CHECK: affine.load {{.*}}[[GETMEM1]]
    %7 = affine.load %0[%c0, %c0, %c0, %3] : memref<10x8x1x8xi8>
    %8 = arith.addi %6, %7 : i8
    // CHECK: arith.constant 22
    %9 = affine.load %2[%c0] : memref<3xi8>
    %10 = arith.addi %8, %9 : i8
    return %10 : i8
  }
}
