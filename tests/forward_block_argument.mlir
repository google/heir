// RUN: heir-opt --unroll-and-forward %s 2>&1 | FileCheck %s

// CHECK: module
module {
  // CHECK: func.func @main
  func.func @main(%arg0: memref<2xi8>) -> memref<2xi8> {
    // CHECK: [[CONST:%[a-z0-9_-]+]] = arith.constant 42
    %c42_i8 = arith.constant 42 : i8
    %c0 = arith.constant 0 : index
    affine.store %c42_i8, %arg0[%c0]: memref<2xi8>

    // CHECK: [[MEM:%[a-z0-9_]+]] = memref.alloc
    // CHECK-COUNT-1: affine.store {{.*}}[[CONST]], {{.*}}[[MEM]]
    // CHECK-COUNT-1: affine.load
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2xi8>
      affine.for %arg2 = 0 to 3 {
            %t = affine.load %arg0[%arg2] : memref<2xi8>
            affine.store %t, %alloc[%arg2] : memref<2xi8>
      }

    // CHECK: return
    return %alloc: memref<2xi8>
  }
}