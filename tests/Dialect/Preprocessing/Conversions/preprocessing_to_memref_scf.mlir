// RUN: heir-opt --preprocessing-to-memref %s | FileCheck %s

// CHECK: func @valid_pairing
// CHECK-SAME: (%[[arg0_val:.*]]: i32) -> i32
func.func @valid_pairing(%arg0: i32) -> i32 {
  // CHECK: %[[storage_val:.*]] = memref.alloc() : memref<7xi32>
  %storage = preprocessing.empty : !preprocessing.storage<i32>
  // CHECK: %[[c0:.*]] = arith.constant 0 : index
  // CHECK: memref.store %[[arg0_val]], %[[storage_val]][%[[c0]]] : memref<7xi32>
  preprocessing.store %arg0, %storage[] site 0 <i32> : i32, !preprocessing.storage<i32>
  // CHECK: %[[c0_1:.*]] = arith.constant 0 : index
  // CHECK: %[[res:.*]] = memref.load %[[storage_val]][%[[c0_1]]] : memref<7xi32>
  %res = preprocessing.load %storage[] site 0 <i32> : !preprocessing.storage<i32>, i32
  return %res : i32
}

// CHECK: func @nested_loops_scf
// CHECK-SAME: (%[[arg0_loop:.*]]: i32)
func.func @nested_loops_scf(%arg0: i32) {
  // CHECK: %[[storage_loop:.*]] = memref.alloc() : memref<7xi32>
  %storage = preprocessing.empty : !preprocessing.storage<i32>
  // CHECK: %[[c0_loop:.*]] = arith.constant 0 : index
  // CHECK: %[[c1_loop:.*]] = arith.constant 1 : index
  // CHECK: %[[c2_loop:.*]] = arith.constant 2 : index
  // CHECK: %[[c3_loop:.*]] = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  // CHECK: scf.for %[[i:.*]] = %[[c0_loop]] to %[[c2_loop]] step %[[c1_loop]] {
  scf.for %i = %c0 to %c2 step %c1 {
    // CHECK: scf.for %[[j:.*]] = %[[c0_loop]] to %[[c3_loop]] step %[[c1_loop]] {
    scf.for %j = %c0 to %c3 step %c1 {
      // CHECK: %[[c1_1:.*]] = arith.constant 1 : index
      // CHECK: %[[addi1:.*]] = arith.addi %[[c1_1]], %[[j]] : index
      // CHECK: %[[c3_1:.*]] = arith.constant 3 : index
      // CHECK: %[[muli:.*]] = arith.muli %[[i]], %[[c3_1]] : index
      // CHECK: %[[addi2:.*]] = arith.addi %[[addi1]], %[[muli]] : index
      // CHECK: memref.store %[[arg0_loop]], %[[storage_loop]][%[[addi2]]] : memref<7xi32>
      preprocessing.store %arg0, %storage[%i, %j] site 1 <i32> : i32, !preprocessing.storage<i32>
    }
  }
  return
}
