// RUN: heir-opt --preprocessing-to-memref %s | FileCheck %s

// CHECK: func @multi_type_lowering
// CHECK-SAME: (%[[arg0:.*]]: i32, %[[arg1:.*]]: f32)
func.func @multi_type_lowering(%arg0: i32, %arg1: f32) {
  // CHECK: %[[storage_i32:.*]] = memref.alloc() : memref<5xi32>
  // CHECK: %[[storage_f32:.*]] = memref.alloc() : memref<1xf32>
  %storage = preprocessing.empty : !preprocessing.storage<i32, f32>

  // Site 0: i32, size 2 (offset 0)
  affine.for %i = 0 to 2 {
    // CHECK: %[[offset0:.*]] = arith.constant 0 : index
    // CHECK: %[[idx0:.*]] = arith.addi %[[offset0]], %{{.*}} : index
    // CHECK: memref.store %[[arg0]], %[[storage_i32]][%[[idx0]]] : memref<5xi32>
    preprocessing.store %arg0, %storage[%i] site 0 <i32> : i32, !preprocessing.storage<i32, f32>
  }

  // Site 1: f32, size 1 (offset 0)
  // CHECK: %[[offset1:.*]] = arith.constant 0 : index
  // CHECK: memref.store %[[arg1]], %[[storage_f32]][%[[offset1]]] : memref<1xf32>
  preprocessing.store %arg1, %storage[] site 1 <f32> : f32, !preprocessing.storage<i32, f32>

  // Site 2: i32, size 3 (offset 2)
  affine.for %j = 0 to 3 {
    // CHECK: %[[offset2:.*]] = arith.constant 2 : index
    // CHECK: %[[idx2:.*]] = arith.addi %[[offset2]], %{{.*}} : index
    // CHECK: memref.store %[[arg0]], %[[storage_i32]][%[[idx2]]] : memref<5xi32>
    preprocessing.store %arg0, %storage[%j] site 2 <i32> : i32, !preprocessing.storage<i32, f32>
  }

  // Load from Site 1 (f32)
  // CHECK: %[[offset1_load:.*]] = arith.constant 0 : index
  // CHECK: %[[res_f32:.*]] = memref.load %[[storage_f32]][%[[offset1_load]]] : memref<1xf32>
  %res_f32 = preprocessing.load %storage[] site 1 <f32> : !preprocessing.storage<i32, f32>, f32

  // Load from Site 2 (i32) inside a loop
  affine.for %k = 0 to 3 {
    // CHECK: %[[offset2_load:.*]] = arith.constant 2 : index
    // CHECK: %[[idx2_load:.*]] = arith.addi %[[offset2_load]], %{{.*}} : index
    // CHECK: %[[res_i32:.*]] = memref.load %[[storage_i32]][%[[idx2_load]]] : memref<5xi32>
    %res_i32 = preprocessing.load %storage[%k] site 2 <i32> : !preprocessing.storage<i32, f32>, i32
  }

  return
}

// The i64 part is never stored to, so it can be omitted.
// CHECK: func @test_unstored_and_multidim
// CHECK-SAME: (%[[arg0_new:.*]]: i16)
func.func @test_unstored_and_multidim(%arg0: i16) {
  // CHECK: %[[storage_i16:.*]] = memref.alloc() : memref<6xi16>
  // CHECK: %[[storage_i64:.*]] = memref.alloc() : memref<0xi64>
  %storage = preprocessing.empty : !preprocessing.storage<i16, i64>

  // Multi-dimensional nested loop (2x3) -> size 6
  // Site 3: i16, size 6 (offset 0)
  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 3 {
      // CHECK: %[[base_offset:.*]] = arith.constant 0 : index
      // CHECK: %[[add_j:.*]] = arith.addi %[[base_offset]], %{{.*}} : index
      // CHECK: %[[stride:.*]] = arith.constant 3 : index
      // CHECK: %[[mul_i:.*]] = arith.muli %{{.*}}, %[[stride]] : index
      // CHECK: %[[final_idx:.*]] = arith.addi %[[add_j]], %[[mul_i]] : index
      // CHECK: memref.store %[[arg0_new]], %[[storage_i16]][%[[final_idx]]] : memref<6xi16>
      preprocessing.store %arg0, %storage[%i, %j] site 3 <i16> : i16, !preprocessing.storage<i16, i64>
    }
  }

  return
}
