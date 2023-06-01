// RUN: heir-opt --unroll-and-forward %s 2>&1 | FileCheck %s

// CHECK: module
module {
  // CHECK: func.func @main
  func.func @main(%arg0: memref<1x80xi8>) -> () {
    // CHECK: [[CONST:%[a-z0-9_-]+]] = arith.constant -128
    %c-128_i8 = arith.constant -128 : i8
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x3x2x1xi8>
      affine.for %arg2 = 0 to 3 {
        affine.for %arg3 = 0 to 2 {
            affine.store %c-128_i8, %alloc[%c0, %arg2, %arg3, %c0] : memref<1x3x2x1xi8>
        }
      }

  // CHECK: [[MEM1:%[a-z0-9_]+]] = memref.alloc
  // CHECK-NOT: affine.load
  // CHECK-COUNT-6: affine.store {{.*}}[[CONST]], {{.*}}[[MEM1]]
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x3x2x1xi8>
      affine.for %arg1 = 0 to 3 {
        affine.for %arg2 = 0 to 2 {
            %12 = affine.load %alloc[%c0, %arg1, %arg2, %c0] : memref<1x3x2x1xi8>
            affine.store %12, %alloc_0[%c0, %arg1, %arg2, %c0] : memref<1x3x2x1xi8>
        }
      }
    return
  }
}