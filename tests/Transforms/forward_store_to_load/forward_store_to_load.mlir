// RUN: heir-opt --forward-store-to-load %s | FileCheck %s

// CHECK-LABEL: func.func @single_store
// CHECK-SAME: %[[ARG0:.*]]: memref<10xi8>
func.func @single_store(%arg0: memref<10xi8>) -> i8 {
  // CHECK-NEXT: %[[VAL:.*]] = arith.constant 7
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: memref.store %[[VAL]], %[[ARG0]][%[[C0]]] : memref<10xi8>
  // CHECK-NEXT: return %[[VAL]] : i8
  %0 = arith.constant 7 : i8
  %c0 = arith.constant 0 : index
  memref.store %0, %arg0[%c0] : memref<10xi8>
  %1 = memref.load %arg0[%c0] : memref<10xi8>
  return %1: i8
}

// CHECK-LABEL: func.func @block_arg
// CHECK-SAME: %[[ARG1:.*]]: i8
func.func @block_arg(%arg0: i8) -> i8 {
  %c0 = arith.constant 0 : index
  %0 = memref.alloc() : memref<1xi8>
  memref.store %arg0, %0[%c0] : memref<1xi8>
  // CHECK-NOT: memref.load
  %1 = memref.load %0[%c0] : memref<1xi8>
  // CHECK: return %[[ARG1]] : i8
  return %1: i8
}


// CHECK-LABEL: func.func @inside_region
// CHECK-SAME: (%[[MEMREF0:.*]]: memref<10xi8>, %[[MEMREF1:.*]]: memref<10xi8>)
func.func @inside_region(%memref0: memref<10xi8>, %memref1: memref<10xi8>) -> memref<10xi8> {
  // CHECK: %[[OUT:.*]] = memref.alloc() : memref<10xi8>
  %out = memref.alloc() : memref<10xi8>
  // CHECK: affine.for %[[I:.*]] = 0 to 10 {
  affine.for %i = 0 to 10 {
    // CHECK: %[[V0:.*]] = memref.load %[[MEMREF0]][%[[I]]]
    %0 = memref.load %memref0[%i] : memref<10xi8>
    // CHECK: memref.store %[[V0]], %[[MEMREF1]][%[[I]]]
    // CHECK-NOT: memref.load
    memref.store %0, %memref1[%i] : memref<10xi8>
    // CHECK-NEXT: memref.store %[[V0]], %[[OUT]][%[[I]]]
    %1 = memref.load %memref1[%i] : memref<10xi8>
    memref.store %1, %out[%i] : memref<10xi8>
  }
  return %out: memref<10xi8>
}

// Two possibilities to forward to, always use the latest
// CHECK-LABEL: func.func @forward_latest
// CHECK-SAME: (%[[MEMREF0:.*]]: memref<10xi8>)
func.func @forward_latest(%memref0: memref<10xi8>) -> i8 {
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // This constant is eliminated.
  %val0 = arith.constant 7 : i8
  // CHECK-NEXT: %[[VAL1:.*]] = arith.constant 8 : i8
  %val1 = arith.constant 8 : i8

  // This next store is unused and eliminated.
  memref.store %val0, %memref0[%c0] : memref<10xi8>
  // CHECK-NEXT: memref.store %[[VAL1]], %[[MEMREF0]][%[[C0]]]
  memref.store %val1, %memref0[%c0] : memref<10xi8>
  // CHECK-NOT: memref.load
  // CHECK-NEXT: return %[[VAL1]] : i8
  %1 = memref.load %memref0[%c0] : memref<10xi8>
  return %1: i8
}


// CHECK-LABEL: func.func @skip_different_blocks
// CHECK-SAME: (%[[MEMREF0:.*]]: memref<10xi8>)
func.func @skip_different_blocks(%memref0: memref<10xi8>) -> i8 {
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-NEXT: %[[VAL0:.*]] = arith.constant 7 : i8
  %val0 = arith.constant 7 : i8
  %true = arith.constant true

  scf.if %true {
    // CHECK: memref.store %[[VAL0]], %[[MEMREF0]][%[[C0]]]
    memref.store %val0, %memref0[%c0] : memref<10xi8>
  }

  // The store is in a different block, so we don't forward it
  // CHECK: %[[V1:.*]] = memref.load %[[MEMREF0]][%[[C0]]]
  %1 = memref.load %memref0[%c0] : memref<10xi8>
  // CHECK-NEXT: return %[[V1]] : i8
  return %1: i8
}


// CHECK-LABEL: func.func @skip_intermediate_region_holding_op
// CHECK-SAME: (%[[MEMREF0:.*]]: memref<10xi8>)
func.func @skip_intermediate_region_holding_op(%memref0: memref<10xi8>) -> i8 {
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-NEXT: %[[VAL0:.*]] = arith.constant 7 : i8
  %val0 = arith.constant 7 : i8
  %true = arith.constant true
  // CHECK: memref.alloc
  %0 = memref.alloc() : memref<1xi8>

  // CHECK: memref.store %[[VAL0]], %[[MEMREF0]][%[[C0]]]
  memref.store %val0, %memref0[%c0] : memref<10xi8>

  // CHECK-NEXT: scf.if
  // CHECK-NEXT: %[[V1:.*]] = memref.load %[[MEMREF0]][%[[C0]]]
  // CHECK-NEXT: memref.store %[[V1]]
  scf.if %true {
    %1 = memref.load %memref0[%c0] : memref<10xi8>
    memref.store %1, %0[%c0] : memref<1xi8>
  }

  // The store has a region-holding op between it and this load, and we don't
  // check if the memref is impacted inside that region. Assume it is and don't
  // forward.
  // CHECK: %[[V2:.*]] = memref.load %[[MEMREF0]][%[[C0]]]
  %2 = memref.load %memref0[%c0] : memref<10xi8>
  // CHECK-NEXT: return %[[V2]] : i8
  return %2: i8
}


// CHECK-LABEL: func.func @wrong_indices
// CHECK-SAME: (%[[MEMREF0:.*]]: memref<10xi8>)
func.func @wrong_indices(%memref0: memref<10xi8>) -> i8 {
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %val0 = arith.constant 7 : i8
  memref.store %val0, %memref0[%c0] : memref<10xi8>

  // Loading at a different index, so do not forward
  // CHECK: %[[V2:.*]] = memref.load %[[MEMREF0]][%[[C1]]]
  %2 = memref.load %memref0[%c1] : memref<10xi8>
  // CHECK-NEXT: return %[[V2]] : i8
  return %2: i8
}

// CHECK-LABEL: func.func @cf_between_stores
// CHECK-SAME: (%[[MEMREF0:.*]]: memref<10xi8>)
func.func @cf_between_stores(%memref0: memref<10xi8>) -> i8 {
  // CHECK-NEXT: %[[TRUE:.*]] = arith.constant true
  %true = arith.constant true
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-NEXT: %[[VAL0:.*]] = arith.constant 7 : i8
  %val0 = arith.constant 7 : i8
  // CHECK-NEXT: %[[VAL1:.*]] = arith.constant 8 : i8
  %val1 = arith.constant 8 : i8

  // The next store is not eliminated because there is a region with control
  // flow between.
  // CHECK-NEXT: memref.store %[[VAL0]], %[[MEMREF0]][%[[C0]]]
  memref.store %val0, %memref0[%c0] : memref<10xi8>

  // CHECK-NEXT: scf.if %[[TRUE]]
  scf.if %true {
    %1 = memref.load %memref0[%c0] : memref<10xi8>
    memref.store %1, %memref0[%c0] : memref<10xi8>
  }

  // CHECK: memref.store %[[VAL1]], %[[MEMREF0]][%[[C0]]]
  memref.store %val1, %memref0[%c0] : memref<10xi8>
  // CHECK-NOT: memref.load
  // CHECK-NEXT: return %[[VAL1]] : i8
  %1 = memref.load %memref0[%c0] : memref<10xi8>
  return %1: i8
}

// CHECK-LABEL: func.func @affine_ops
// CHECK-SAME: %[[ARG1:.*]]: i8, %[[ARG2:.*]]: i8
func.func @affine_ops(%arg0: i8, %arg1: i8) -> (i8, i8) {
  %c0 = arith.constant 0 : index
  %0 = memref.alloc() : memref<1xi8>
  affine.store %arg0, %0[%c0] : memref<1xi8>
  // CHECK-NOT: memref.load
  %1 = affine.load %0[%c0] : memref<1xi8>
  affine.store %arg1, %0[%c0] : memref<1xi8>
  %3 = affine.load %0[%c0] : memref<1xi8>
  // CHECK: return %[[ARG1]], %[[ARG2]] : i8, i8
  return %1, %3 : i8, i8
}
