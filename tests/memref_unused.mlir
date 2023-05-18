// b/281566825: Re-enable test when pipeline is complete.
// XFAIL: *
// RUN: heir-opt --memref2arith %s | FileCheck --check-prefix=CHECK_ERROR_1 %s

// Note: This verifies that the memref allocation is removed, along with its
//   referencing affine store.

// CHECK-LABEL: func @store_only_affine
func.func @store_only_affine() -> i32 {
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: affine.store
  // CHECK: return

  %c_42 = arith.constant 42 : i32
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<1x1xi32>
  affine.store %c_42, %alloc[%c0, %c0] :  memref<1x1xi32>
  %v1 = arith.addi %c_42, %c_42 : i32
  return %v1 : i32
}