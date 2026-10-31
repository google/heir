// RUN: heir-opt %s --cheddar-gpu-to-gpu --gpu-kernel-outlining --mod-arith-to-arith | FileCheck %s

// CHECK-NOT: cheddar_gpu.mod_down_epilogue
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.remsi

module attributes {
  cheddar_gpu.params = #cheddar_gpu.params<
    word = 32,
    log_n = 4,
    ter = [],
    main = [974258177, 1081212929, 1196556289, 993263617],
    aux = [2147352577, 2146959361],
    level_config = [[[2 : i32, 0 : i32]]],
    alpha = 2,
    dnum = 0
  >
} {
  // CHECK: gpu.module @cheddar_kernels
  // CHECK: gpu.func @cheddar_gpu_mod_down_epilogue_

  // CHECK: func.func @test_epilogue_basic
  // CHECK-NOT: cheddar_gpu.mod_down_epilogue
  // CHECK: gpu.launch_func
  func.func @test_epilogue_basic(
      %out: memref<2x16xi32>,
      %ntt: memref<2x16xi32>,
      %inv: memref<2xi32>,
      %src2: memref<2x16xi32>) {
    cheddar_gpu.mod_down_epilogue %out, %ntt, %inv, %src2 <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>,
      pad_start = 0 : i64,
      pad_end = 2 : i64
    }> : memref<2x16xi32>, memref<2x16xi32>, memref<2xi32>, memref<2x16xi32>
    return
  }

  // CHECK: func.func @test_epilogue_padding
  // CHECK-NOT: cheddar_gpu.mod_down_epilogue
  // CHECK: gpu.launch_func
  func.func @test_epilogue_padding(
      %out: memref<2x16xi32>,
      %ntt: memref<2x16xi32>,
      %inv: memref<2xi32>,
      %src2: memref<2x16xi32>,
      %pad: memref<2xi32>) {
    cheddar_gpu.mod_down_epilogue %out, %ntt, %inv, %src2, %pad <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>,
      pad_start = 0 : i64,
      pad_end = 2 : i64
    }> : memref<2x16xi32>, memref<2x16xi32>, memref<2xi32>, memref<2x16xi32>, memref<2xi32>
    return
  }

  // CHECK: func.func @test_epilogue_no_src2
  // CHECK-NOT: cheddar_gpu.mod_down_epilogue
  // CHECK: gpu.launch_func
  func.func @test_epilogue_no_src2(
      %out: memref<2x16xi32>,
      %ntt: memref<2x16xi32>,
      %inv: memref<2xi32>) {
    cheddar_gpu.mod_down_epilogue %out, %ntt, %inv <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>,
      pad_start = 0 : i64,
      pad_end = 0 : i64
    }> : memref<2x16xi32>, memref<2x16xi32>, memref<2xi32>
    return
  }

  memref.global "private" constant @__inv_prime_prod : memref<2xi32> = dense<1>
  memref.global "private" constant @__src2_padding : memref<2xi32> = dense<1>

  // CHECK: func.func @test_epilogue_device_residency
  // CHECK: call @__cheddar_gpu_init()
  // CHECK: llvm.mlir.addressof @__cheddar_dev_ptr___inv_prime_prod
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_mod_down_epilogue_src2_pad_w32_deg16
  // CHECK: return
  func.func @test_epilogue_device_residency(
      %ntt: memref<2x16xi32>,
      %src2: memref<2x16xi32>) {
    %inv = memref.get_global @__inv_prime_prod : memref<2xi32>
    %pad = memref.get_global @__src2_padding : memref<2xi32>
    %out = memref.alloc() : memref<2x16xi32>
    cheddar_gpu.mod_down_epilogue %out, %ntt, %inv, %src2, %pad <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>,
      pad_start = 0 : i64,
      pad_end = 2 : i64
    }> : memref<2x16xi32>, memref<2x16xi32>, memref<2xi32>, memref<2x16xi32>, memref<2xi32>
    memref.dealloc %out : memref<2x16xi32>
    return
  }
}
