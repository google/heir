// RUN: heir-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" | FileCheck %s --check-prefix=IDENTITY
// RUN: heir-opt %s --one-shot-bufferize="bufferize-function-boundaries" | FileCheck %s --check-prefix=DYNAMIC

// cheddar_gpu.add requires out, lhs, and rhs to have the same type. When their
// buffers disagree on layout, bufferization casts them to the fully dynamic
// layout instead of creating an op that fails verification.

#eval = #cheddar_gpu.form<eval>

// A fresh allocation has an identity layout. With identity-layout function
// arguments nothing needs a cast.
// IDENTITY: func.func @test_add_fresh_init
// IDENTITY: %[[OUT:.*]] = memref.alloc()
// IDENTITY-NOT: memref.cast
// IDENTITY: cheddar_gpu.add %[[OUT]], %{{.*}}, %{{.*}} : memref<4x65536xi32>{{$}}

// With the default fully dynamic layout for function arguments, only the
// allocation is cast, and the uncast allocation is what gets returned.
// DYNAMIC: func.func @test_add_fresh_init
// DYNAMIC-SAME: (%[[LHS:.*]]: memref<4x65536xi32, strided<[?, ?], offset: ?>>, %[[RHS:.*]]: memref<4x65536xi32, strided<[?, ?], offset: ?>>)
// DYNAMIC: %[[OUT:.*]] = memref.alloc()
// DYNAMIC: %[[OUT_CAST:.*]] = memref.cast %[[OUT]] : memref<4x65536xi32> to memref<4x65536xi32, strided<[?, ?], offset: ?>>
// DYNAMIC: cheddar_gpu.add %[[OUT_CAST]], %[[LHS]], %[[RHS]]
// DYNAMIC-SAME: : memref<4x65536xi32, strided<[?, ?], offset: ?>>
// DYNAMIC: return %[[OUT]] : memref<4x65536xi32>
func.func @test_add_fresh_init(%lhs: tensor<4x65536xi32>, %rhs: tensor<4x65536xi32>) -> tensor<4x65536xi32> {
  %empty = tensor.empty() : tensor<4x65536xi32>
  %0 = cheddar_gpu.add %empty, %lhs, %rhs <{
    lhs_form = #eval,
    rhs_form = #eval,
    out_form = #eval
  }> : tensor<4x65536xi32> -> tensor<4x65536xi32>
  return %0 : tensor<4x65536xi32>
}

// A slice bufferizes to a subview with a strided layout, which differs from
// the identity layout of the other two buffers, so all three are cast.
// IDENTITY: func.func @test_add_slice_input
// IDENTITY-SAME: (%[[BIG:.*]]: memref<8x65536xi32>, %[[RHS:.*]]: memref<4x65536xi32>)
// IDENTITY: %[[LHS:.*]] = memref.subview %[[BIG]]
// IDENTITY: %[[OUT:.*]] = memref.alloc()
// IDENTITY: %[[OUT_CAST:.*]] = memref.cast %[[OUT]] : memref<4x65536xi32> to memref<4x65536xi32, strided<[?, ?], offset: ?>>
// IDENTITY: %[[LHS_CAST:.*]] = memref.cast %[[LHS]] : memref<4x65536xi32, strided<[65536, 1], offset: 262144>> to memref<4x65536xi32, strided<[?, ?], offset: ?>>
// IDENTITY: %[[RHS_CAST:.*]] = memref.cast %[[RHS]] : memref<4x65536xi32> to memref<4x65536xi32, strided<[?, ?], offset: ?>>
// IDENTITY: cheddar_gpu.add %[[OUT_CAST]], %[[LHS_CAST]], %[[RHS_CAST]]
// IDENTITY-SAME: : memref<4x65536xi32, strided<[?, ?], offset: ?>>
// IDENTITY: return %[[OUT]] : memref<4x65536xi32>
func.func @test_add_slice_input(%big: tensor<8x65536xi32>, %rhs: tensor<4x65536xi32>) -> tensor<4x65536xi32> {
  %lhs = tensor.extract_slice %big[4, 0] [4, 65536] [1, 1] : tensor<8x65536xi32> to tensor<4x65536xi32>
  %empty = tensor.empty() : tensor<4x65536xi32>
  %0 = cheddar_gpu.add %empty, %lhs, %rhs <{
    lhs_form = #eval,
    rhs_form = #eval,
    out_form = #eval
  }> : tensor<4x65536xi32> -> tensor<4x65536xi32>
  return %0 : tensor<4x65536xi32>
}
