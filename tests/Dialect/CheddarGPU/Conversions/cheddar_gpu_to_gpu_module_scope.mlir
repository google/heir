// RUN: heir-opt %s --cheddar-gpu-to-gpu --split-input-file | FileCheck %s

// A module that carries params but has no cheddar_gpu ops is not given a
// kernel module or a prime table.

#params = #cheddar_gpu.params<word = 32, log_n = 8, ter = [17], main = [97], aux = [257], level_config = [[[1 : i32, 1 : i32]]], alpha = 1, dnum = 0>

// CHECK-NOT: gpu.container_module
// CHECK-NOT: gpu.module
// CHECK-NOT: @__cheddar_primes
// CHECK: func.func @test_no_cheddar_gpu_ops
module attributes {cheddar_gpu.params = #params} {
  func.func @test_no_cheddar_gpu_ops() {
    return
  }
}

// -----

// The kernels and the prime table are emitted into the nearest enclosing
// module, which is the one whose params the op is lowered against. The params
// attribute is removed from that module as well as from the top-level one.

#eval = #cheddar_gpu.form<eval>
#params = #cheddar_gpu.params<word = 32, log_n = 8, ter = [17], main = [97], aux = [257], level_config = [[[1 : i32, 1 : i32]]], alpha = 1, dnum = 0>

// CHECK: {{^}}module {
// CHECK-NEXT: module attributes {gpu.container_module} {
// CHECK-NEXT: gpu.module @cheddar_kernels
// CHECK-NEXT: memref.global "private" constant @__cheddar_primes : memref<3xi32> = dense<[17, 97, 257]>
// CHECK-NEXT: gpu.func @cheddar_gpu_add_w32_deg256
// CHECK: func.func @test_nested_module
// CHECK: gpu.launch_func @cheddar_kernels::@cheddar_gpu_add_w32_deg256
module {
  module attributes {cheddar_gpu.params = #params} {
    func.func @test_nested_module(%out: memref<1x256xi32>, %lhs: memref<1x256xi32>, %rhs: memref<1x256xi32>) {
      cheddar_gpu.add %out, %lhs, %rhs <{
        lhs_form = #eval,
        rhs_form = #eval,
        out_form = #eval
      }> : memref<1x256xi32>
      return
    }
  }
}
