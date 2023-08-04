// RUN: heir-opt --expand-copy %s | FileCheck %s

// This verifies that --expand-copy removes memref.copy and rewrites with affine
// loads and stores.

// CHECK-LABEL: func @memref_copy
func.func @memref_copy() -> i32 {
  %c_42 = arith.constant 42 : i32
  %c0 = arith.constant 0 : index
  // CHECK: [[MEM1:%[a-z0-9_]+]] = memref.alloc
  %alloc = memref.alloc() : memref<1x1xi32>
  // CHECK: [[MEM2:%[a-z0-9_]+]] = memref.alloc
  %alloc0 = memref.alloc() : memref<1x1xi32>
  affine.store %c_42, %alloc[%c0, %c0] :  memref<1x1xi32>
  // CHECK-NOT: memref.copy
  // CHECK: affine.load {{.*}}[[MEM1]]
  // CHECK-NEXT: affine.store {{.*}}[[MEM2]]
  memref.copy %alloc, %alloc0 : memref<1x1xi32> to memref<1x1xi32>
  %v1 = arith.addi %c_42, %c_42 : i32
  // CHECK: return
  return %v1 : i32
}
