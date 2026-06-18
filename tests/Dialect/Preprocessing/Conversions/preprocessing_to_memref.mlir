// RUN: heir-opt --preprocessing-to-memref %s | FileCheck %s

// CHECK: func @valid_pairing
// CHECK-SAME: (%[[arg0:.*]]: i32) -> i32
func.func @valid_pairing(%arg0: i32) -> i32 {
  // CHECK: %[[storage:.*]] = memref.alloc() : memref<7xi32>
  %storage = preprocessing.empty : !preprocessing.storage<i32>
  // CHECK: %[[c0:.*]] = arith.constant 0 : index
  // CHECK: memref.store %[[arg0]], %[[storage]][%[[c0]]] : memref<7xi32>
  preprocessing.store %arg0, %storage[] site 0 <i32> : i32, !preprocessing.storage<i32>
  // CHECK: %[[c0_1:.*]] = arith.constant 0 : index
  // CHECK: %[[res:.*]] = memref.load %[[storage]][%[[c0_1]]] : memref<7xi32>
  %res = preprocessing.load %storage[] site 0 <i32> : !preprocessing.storage<i32>, i32
  return %res : i32
}

// CHECK: func @nested_loops
// CHECK-SAME: (%[[arg0:.*]]: i32)
func.func @nested_loops(%arg0: i32) {
  // CHECK: %[[storage:.*]] = memref.alloc() : memref<7xi32>
  %storage = preprocessing.empty : !preprocessing.storage<i32>
  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 3 {
      // CHECK: %[[c1:.*]] = arith.constant 1 : index
      // CHECK: %[[addi1:.*]] = arith.addi %[[c1]], %{{.*}} : index
      // CHECK: %[[c3:.*]] = arith.constant 3 : index
      // CHECK: %[[muli:.*]] = arith.muli %{{.*}}, %[[c3]] : index
      // CHECK: %[[addi2:.*]] = arith.addi %[[addi1]], %[[muli]] : index
      // CHECK: memref.store %[[arg0]], %[[storage]][%[[addi2]]] : memref<7xi32>
      preprocessing.store %arg0, %storage[%i, %j] site 1 <i32> : i32, !preprocessing.storage<i32>
    }
  }
  return
}

// CHECK: func @func_sig
// CHECK-SAME: (%[[arg0:.*]]: memref<7xi32>) -> memref<7xi32>
func.func @func_sig(%arg0: !preprocessing.storage<i32>) -> !preprocessing.storage<i32> {
  return %arg0 : !preprocessing.storage<i32>
}
