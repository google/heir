// RUN: heir-opt --preprocessing-to-memref %s | FileCheck %s

// Use a dummy plaintext type
!pt = i32

// CHECK: func @store_stuff
// CHECK-SAME: (%[[arg0:.*]]: i32) -> memref<11xi32>
func.func @store_stuff(%arg0: !pt) -> !preprocessing.storage<!pt> {
  // CHECK: %[[storage:.*]] = memref.alloc() : memref<11xi32>
  %storage = preprocessing.empty : !preprocessing.storage<!pt>

  // Out of loop, site 0
  // CHECK: %[[c0:.*]] = arith.constant 0 : index
  // CHECK: memref.store %[[arg0]], %[[storage]][%[[c0]]] : memref<11xi32>
  preprocessing.store %arg0, %storage[] site 0 <!pt> : !pt, !preprocessing.storage<!pt>

  // In a loop, site 1
  // CHECK: affine.for %[[i:.*]] = 0 to 4 {
  affine.for %i = 0 to 4 {
    // CHECK: %[[c1:.*]] = arith.constant 1 : index
    // CHECK: %[[idx1:.*]] = arith.addi %[[c1]], %[[i]] : index
    // CHECK: memref.store %[[arg0]], %[[storage]][%[[idx1]]] : memref<11xi32>
    preprocessing.store %arg0, %storage[%i] site 1 <!pt> : !pt, !preprocessing.storage<!pt>
  }

  // In a doubly-nested loop, site 2
  // CHECK: affine.for %[[j:.*]] = 0 to 2 {
  affine.for %j = 0 to 2 {
    // CHECK: affine.for %[[k:.*]] = 0 to 3 {
    affine.for %k = 0 to 3 {
      // Offset for site 2 is 1 + 4 = 5.
      // CHECK: %[[c5:.*]] = arith.constant 5 : index
      // CHECK: %[[addi1:.*]] = arith.addi %[[c5]], %[[k]] : index
      // CHECK: %[[c3:.*]] = arith.constant 3 : index
      // CHECK: %[[muli:.*]] = arith.muli %[[j]], %[[c3]] : index
      // CHECK: %[[idx2:.*]] = arith.addi %[[addi1]], %[[muli]] : index
      // CHECK: memref.store %[[arg0]], %[[storage]][%[[idx2]]] : memref<11xi32>
      preprocessing.store %arg0, %storage[%j, %k] site 2 <!pt> : !pt, !preprocessing.storage<!pt>
    }
  }

  // CHECK: return %[[storage]] : memref<11xi32>
  return %storage : !preprocessing.storage<!pt>
}

// CHECK: func @load_stuff
// CHECK-SAME: (%[[storage_arg:.*]]: memref<11xi32>)
func.func @load_stuff(%storage: !preprocessing.storage<!pt>) {
  // Out of loop, site 0
  // CHECK: %[[c0_load:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = memref.load %[[storage_arg]][%[[c0_load]]] : memref<11xi32>
  %res0 = preprocessing.load %storage[] site 0 <!pt> : !preprocessing.storage<!pt>, !pt

  // In a loop, site 1
  // CHECK: affine.for %[[i_load:.*]] = 0 to 4 {
  affine.for %i = 0 to 4 {
    // CHECK: %[[c1_load:.*]] = arith.constant 1 : index
    // CHECK: %[[idx1_load:.*]] = arith.addi %[[c1_load]], %[[i_load]] : index
    // CHECK: %{{.*}} = memref.load %[[storage_arg]][%[[idx1_load]]] : memref<11xi32>
    %res1 = preprocessing.load %storage[%i] site 1 <!pt> : !preprocessing.storage<!pt>, !pt
  }

  // In a doubly-nested loop, site 2
  // CHECK: affine.for %[[j_load:.*]] = 0 to 2 {
  affine.for %j = 0 to 2 {
    // CHECK: affine.for %[[k_load:.*]] = 0 to 3 {
    affine.for %k = 0 to 3 {
      // CHECK: %[[c5_load:.*]] = arith.constant 5 : index
      // CHECK: %[[addi1_load:.*]] = arith.addi %[[c5_load]], %[[k_load]] : index
      // CHECK: %[[c3_load:.*]] = arith.constant 3 : index
      // CHECK: %[[muli_load:.*]] = arith.muli %[[j_load]], %[[c3_load]] : index
      // CHECK: %[[idx2_load:.*]] = arith.addi %[[addi1_load]], %[[muli_load]] : index
      // CHECK: %{{.*}} = memref.load %[[storage_arg]][%[[idx2_load]]] : memref<11xi32>
      %res2 = preprocessing.load %storage[%j, %k] site 2 <!pt> : !preprocessing.storage<!pt>, !pt
    }
  }

  return
}
