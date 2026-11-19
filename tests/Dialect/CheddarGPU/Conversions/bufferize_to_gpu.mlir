// RUN: heir-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cheddar-gpu-to-gpu | FileCheck %s

#eval = #cheddar_gpu.form<eval>
#coeff = #cheddar_gpu.form<coeff>
#params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [17], main = [97, 193], aux = [257], level_config = [[[2 : i32, 1 : i32]]], alpha = 1, dnum = 0>

module attributes {
  cheddar_gpu.params = #params
} {
  // (a) Empty tensor init: bufferizes with exactly one alloc, no copy, and one launch
  // CHECK: func.func @test_add_empty
  // CHECK: %[[ALLOC:.*]] = memref.alloc()
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.copy
  // CHECK: gpu.launch_func @cheddar_kernels::@cheddar_gpu_add_w32_deg65536
  func.func @test_add_empty(%lhs: tensor<4x65536xi32>, %rhs: tensor<4x65536xi32>) -> tensor<4x65536xi32> {
    %empty = tensor.empty() : tensor<4x65536xi32>
    %0 = cheddar_gpu.add %empty, %lhs, %rhs <{
      lhs_form = #eval,
      rhs_form = #eval,
      out_form = #eval
    }> : tensor<4x65536xi32> -> tensor<4x65536xi32>
    return %0 : tensor<4x65536xi32>
  }

  // (b) In-place add: bufferizes with no allocs and lowers to gpu.launch_func
  // CHECK: func.func @test_add_inplace
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.copy
  // CHECK: gpu.launch_func @cheddar_kernels::@cheddar_gpu_add_w32_deg65536
  func.func @test_add_inplace(%out: tensor<4x65536xi32>, %lhs: tensor<4x65536xi32>, %rhs: tensor<4x65536xi32>) -> tensor<4x65536xi32> {
    %0 = cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #eval,
      rhs_form = #eval,
      out_form = #eval
    }> : tensor<4x65536xi32> -> tensor<4x65536xi32>
    return %0 : tensor<4x65536xi32>
  }

  // (c) Out-of-place add: reading %lhs after using it as init causes an RaW conflict,
  // prompting One-Shot Bufferize to insert a buffer allocation.
  // Because readsDpsInit is false (the add writes to %out without reading its initial content),
  // no memref.copy is needed into the newly allocated buffer.
  // CHECK: func.func @test_add_outofplace
  // CHECK: memref.alloc
  // CHECK-NOT: memref.copy
  // CHECK: gpu.launch_func @cheddar_kernels::@cheddar_gpu_add_w32_deg65536
  func.func @test_add_outofplace(%lhs: tensor<4x65536xi32>, %rhs: tensor<4x65536xi32>) -> (tensor<4x65536xi32>, tensor<4x65536xi32>) {
    %0 = cheddar_gpu.add %lhs, %lhs, %rhs <{
      lhs_form = #eval,
      rhs_form = #eval,
      out_form = #eval
    }> : tensor<4x65536xi32> -> tensor<4x65536xi32>
    return %0, %lhs : tensor<4x65536xi32>, tensor<4x65536xi32>
  }

  // (d) Chained add: two consecutive additions bufferize and lower to two kernel launches
  // with complete in-place buffer reuse (no allocs or copies).
  // CHECK: func.func @test_add_chained
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.copy
  // CHECK: gpu.launch_func @cheddar_kernels::@cheddar_gpu_add_w32_deg65536
  // CHECK: gpu.launch_func @cheddar_kernels::@cheddar_gpu_add_w32_deg65536
  func.func @test_add_chained(%out: tensor<4x65536xi32>, %a: tensor<4x65536xi32>, %b: tensor<4x65536xi32>, %c: tensor<4x65536xi32>) -> tensor<4x65536xi32> {
    %0 = cheddar_gpu.add %out, %a, %b <{
      lhs_form = #eval,
      rhs_form = #eval,
      out_form = #eval
    }> : tensor<4x65536xi32> -> tensor<4x65536xi32>
    %1 = cheddar_gpu.add %0, %0, %c <{
      lhs_form = #eval,
      rhs_form = #eval,
      out_form = #eval
    }> : tensor<4x65536xi32> -> tensor<4x65536xi32>
    return %1 : tensor<4x65536xi32>
  }
}
