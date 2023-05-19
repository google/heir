// b/281566825: Re-enable test when pipeline is complete.
// XFAIL: *
// RUN: heir-opt --memref2arith %s | FileCheck %s

// Note: this verifies that the MemrefToArith pipeline does not remove any
//   memref allocations and operations that are still in use (due to the copy).
//   TODO(b/281566825): When all patterns are implemented, these memrefs should
//   be removed.

// CHECK-LABEL: func @memref_copy
func.func @memref_copy() -> i32 {
  // CHECK: memref.alloc
  // CHECK: memref.alloc
  // CHECK: return

  %c_42 = arith.constant 42 : i32
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<1x1xi32>
  %alloc0 = memref.alloc() : memref<1x1xi32>
  affine.store %c_42, %alloc[%c0, %c0] :  memref<1x1xi32>
  memref.copy %alloc, %alloc0 : memref<1x1xi32> to memref<1x1xi32>
  %v1 = arith.addi %c_42, %c_42 : i32
  return %v1 : i32
}