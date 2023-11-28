// RUN: heir-opt --unroll-and-forward %s 2>&1 | FileCheck %s

// CHECK: module
module {
  // CHECK: func.func @main
  func.func @main(%arg0: memref<1x80xi8>) -> memref<1x3x2x1xi8> {
    // CHECK: [[CONST1:%[a-z0-9_-]+]] = arith.constant -128
    %c-128_i8 = arith.constant -128 : i8
    // CHECK: [[CONST2:%[a-z0-9_-]+]] = arith.constant 127
    %c127_i8 = arith.constant 127 : i8
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x3x2x1xi8>
      affine.for %arg2 = 0 to 3 {
        affine.for %arg3 = 0 to 2 {
            affine.store %c-128_i8, %alloc[%c0, %arg2, %arg3, %c0] : memref<1x3x2x1xi8>
        }
      }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x3x2x1xi8>
      affine.for %arg1 = 0 to 3 {
        affine.for %arg2 = 0 to 2 {
            %12 = affine.load %alloc[%c0, %arg1, %arg2, %c0] : memref<1x3x2x1xi8>
            affine.store %12, %alloc_1[%c0, %arg1, %arg2, %c0] : memref<1x3x2x1xi8>
        }
      }

  // This updates the values in alloc in the latter half of MEM1.
      affine.for %arg1 = 0 to 3 {
        affine.for %arg2 = 0 to 1 {
            affine.store %c127_i8, %alloc[%c0, %arg1, %arg2, %c0] : memref<1x3x2x1xi8>
        }
      }

  // CHECK: [[MEM2:%[a-z0-9_]+]] = memref.alloc
  // CHECK-NOT: affine.load
  // CHECK-COUNT-3: affine.store {{.*}}[[CONST2]], {{.*}}[[MEM2]]
  // CHECK-COUNT-3: affine.store {{.*}}[[CONST1]], {{.*}}[[MEM2]]
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x3x2x1xi8>
      affine.for %arg2 = 0 to 2 {
        affine.for %arg1 = 0 to 3 {
            %12 = affine.load %alloc[%c0, %arg1, %arg2, %c0] : memref<1x3x2x1xi8>
            affine.store %12, %alloc_2[%c0, %arg1, %arg2, %c0] : memref<1x3x2x1xi8>
        }
      }
    return %alloc_2 : memref<1x3x2x1xi8>
  }
}
