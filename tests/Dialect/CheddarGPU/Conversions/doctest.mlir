// RUN: heir-opt %s --cheddar-gpu-to-gpu | FileCheck %s

#eval = #cheddar_gpu.form<eval>
#params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [17], main = [97, 193], aux = [257], level_config = [[[2 : i32, 1 : i32]]], alpha = 1, dnum = 0>

module attributes {cheddar_gpu.params = #params} {
  // CHECK: func.func @add_example
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[GRID:.*]] = arith.constant 1024 : index
  // CHECK: %[[BLOCK:.*]] = arith.constant 256 : index
  // CHECK: gpu.launch_func @cheddar_kernels::@cheddar_gpu_add_w32_deg65536 blocks in (%[[GRID]], %[[C1]], %[[C1]]) threads in (%[[BLOCK]], %[[C1]], %[[C1]])
  func.func @add_example(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
    cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #eval,
      rhs_form = #eval,
      out_form = #eval,
      num_aux = 1 : i64
    }> : memref<4x65536xi32>
    return
  }
}
