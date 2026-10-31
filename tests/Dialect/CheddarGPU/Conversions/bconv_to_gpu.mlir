// RUN: heir-opt %s --cheddar-gpu-to-gpu --gpu-kernel-outlining --mod-arith-to-arith | FileCheck %s

// CHECK-NOT: cheddar_gpu.bconv
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.remsi

module attributes {
  cheddar_gpu.params = #cheddar_gpu.params<
    word = 32,
    log_n = 4,
    ter = [],
    main = [974258177, 1081212929, 1196556289, 993263617],
    aux = [2147352577, 2146959361, 2146041857],
    level_config = [[[1 : i32, 1 : i32]]],
    alpha = 1,
    dnum = 0
  >
} {
  // CHECK: gpu.module @cheddar_kernels
  // CHECK: gpu.func @cheddar_gpu_bconv_w32_deg16

  // CHECK: func.func @test_bconv_basic
  // CHECK-NOT: cheddar_gpu.bconv
  // CHECK: gpu.launch_func
  func.func @test_bconv_basic(%out: memref<7x16xi32>, %src: memref<4x16xi32>, %table: memref<3x4xi32>) {
    cheddar_gpu.bconv %out, %src, %table <{
      in_form = #cheddar_gpu.form<coeff_centered>,
      out_form = #cheddar_gpu.form<coeff>,
      out_limb_range = array<i64: 4, 3>
    }> : memref<7x16xi32>, memref<4x16xi32>, memref<3x4xi32>
    return
  }

  // CHECK: func.func @test_bconv_skip_range
  // CHECK-NOT: cheddar_gpu.bconv
  // CHECK: gpu.launch_func
  func.func @test_bconv_skip_range(%out: memref<7x16xi32>, %src: memref<4x16xi32>, %table: memref<2x4xi32>) {
    cheddar_gpu.bconv %out, %src, %table <{
      in_form = #cheddar_gpu.form<coeff_centered>,
      out_form = #cheddar_gpu.form<coeff>,
      skip_range = array<i64: 1, 2>,
      out_limb_range = array<i64: 0, 3>
    }> : memref<7x16xi32>, memref<4x16xi32>, memref<2x4xi32>
    return
  }

  memref.global "private" constant @__bconv_table : memref<3x4xi32> = dense<1>

  // CHECK: func.func @test_bconv_device_residency
  // CHECK: call @__cheddar_gpu_init()
  // CHECK: llvm.mlir.addressof @__cheddar_dev_ptr___bconv_table
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_bconv_w32_deg16
  // CHECK: return
  func.func @test_bconv_device_residency(%src: memref<4x16xi32>) {
    %table = memref.get_global @__bconv_table : memref<3x4xi32>
    %out = memref.alloc() : memref<7x16xi32>
    cheddar_gpu.bconv %out, %src, %table <{
      in_form = #cheddar_gpu.form<coeff_centered>,
      out_form = #cheddar_gpu.form<coeff>,
      out_limb_range = array<i64: 4, 3>
    }> : memref<7x16xi32>, memref<4x16xi32>, memref<3x4xi32>
    memref.dealloc %out : memref<7x16xi32>
    return
  }
}
