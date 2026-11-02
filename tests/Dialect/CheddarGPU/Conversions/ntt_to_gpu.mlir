// RUN: heir-opt %s --cheddar-gpu-to-gpu=ntt-impl=simple --gpu-kernel-outlining --mod-arith-to-arith | FileCheck %s

// CHECK-NOT: cheddar_gpu.ntt
// CHECK-NOT: cheddar_gpu.intt
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.remsi

module attributes {
  cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 4, ter = [30539777, 37224449, 33292289, 32899073], main = [], aux = [], level_config = [[[1 : i32, 1 : i32]]], alpha = 1, dnum = 0>
} {
  // CHECK: gpu.module @cheddar_kernels
  // CHECK-DAG: gpu.func @cheddar_gpu_ntt_fwd_stage0_w32_deg16
  // CHECK-DAG: gpu.func @cheddar_gpu_ntt_fwd_stage1_w32_deg16
  // CHECK-DAG: gpu.func @cheddar_gpu_ntt_fwd_stage2_w32_deg16
  // CHECK-DAG: gpu.func @cheddar_gpu_ntt_fwd_stage3_w32_deg16
  // CHECK-DAG: gpu.func @cheddar_gpu_ntt_fwd_mont_w32_deg16
  // CHECK-DAG: gpu.func @cheddar_gpu_intt_inv_stage0_w32_deg16
  // CHECK-DAG: gpu.func @cheddar_gpu_intt_epilogue_none_w32_deg16
  // CHECK-DAG: gpu.func @cheddar_gpu_intt_epilogue_mult_const_w32_deg16
  // CHECK-DAG: gpu.func @cheddar_gpu_intt_epilogue_mult_const_normalize_w32_deg16

  // CHECK: func.func @test_ntt
  // CHECK-NOT: cheddar_gpu.ntt
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_ntt_fwd_stage0_w32_deg16
  func.func @test_ntt(%out: memref<4x16xi32>, %in: memref<4x16xi32>) {
    cheddar_gpu.ntt %out, %in <{
      in_form = #cheddar_gpu.form<coeff>,
      out_form = #cheddar_gpu.form<eval>,
      montgomery_conversion = false,
      out_limb_range = array<i64: 2, 2>
    }> : memref<4x16xi32>
    return
  }

  // CHECK: func.func @test_ntt_mont
  // CHECK-NOT: cheddar_gpu.ntt
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_ntt_fwd_mont_w32_deg16
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_ntt_fwd_stage0_w32_deg16
  func.func @test_ntt_mont(%out: memref<4x16xi32>, %in: memref<4x16xi32>) {
    cheddar_gpu.ntt %out, %in <{
      in_form = #cheddar_gpu.form<coeff>,
      out_form = #cheddar_gpu.form<eval>,
      montgomery_conversion = true,
      out_limb_range = array<i64: 2, 2>
    }> : memref<4x16xi32>
    return
  }

  // CHECK: func.func @test_intt_none
  // CHECK-NOT: cheddar_gpu.intt
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_intt_inv_stage0_w32_deg16
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_intt_epilogue_none_w32_deg16
  func.func @test_intt_none(%out: memref<4x16xi32>, %in: memref<4x16xi32>) {
    cheddar_gpu.intt %out, %in <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<coeff>,
      montgomery_conversion = true,
      epilogue = #cheddar_gpu.intt_epilogue<none>,
      out_limb_range = array<i64: 2, 2>
    }> : memref<4x16xi32>
    return
  }

  // CHECK: func.func @test_intt_none_mont_false
  // CHECK-NOT: cheddar_gpu.intt
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_intt_inv_stage0_w32_deg16
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_intt_epilogue_none_w32_deg16
  func.func @test_intt_none_mont_false(%out: memref<4x16xi32>, %in: memref<4x16xi32>) {
    cheddar_gpu.intt %out, %in <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<coeff>,
      montgomery_conversion = false,
      epilogue = #cheddar_gpu.intt_epilogue<none>,
      out_limb_range = array<i64: 2, 2>
    }> : memref<4x16xi32>
    return
  }

  // CHECK: func.func @test_intt_mult_const
  // CHECK-NOT: cheddar_gpu.intt
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_intt_inv_stage0_w32_deg16
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_intt_epilogue_mult_const_w32_deg16
  func.func @test_intt_mult_const(%out: memref<4x16xi32>, %in: memref<4x16xi32>, %cst: memref<2xi32>) {
    cheddar_gpu.intt %out, %in, %cst <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<coeff>,
      montgomery_conversion = true,
      epilogue = #cheddar_gpu.intt_epilogue<mult_const>,
      out_limb_range = array<i64: 2, 2>
    }> : memref<4x16xi32>, memref<2xi32>
    return
  }

  // CHECK: func.func @test_intt_mult_const_normalize
  // CHECK-NOT: cheddar_gpu.intt
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_intt_inv_stage0_w32_deg16
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_intt_epilogue_mult_const_normalize_w32_deg16
  func.func @test_intt_mult_const_normalize(%out: memref<4x16xi32>, %in: memref<4x16xi32>, %cst: memref<2xi32>) {
    cheddar_gpu.intt %out, %in, %cst <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<coeff_centered>,
      montgomery_conversion = true,
      epilogue = #cheddar_gpu.intt_epilogue<mult_const_normalize>,
      out_limb_range = array<i64: 2, 2>
    }> : memref<4x16xi32>, memref<2xi32>
    return
  }

  // CHECK: func.func @test_ntt_modup
  // CHECK-NOT: cheddar_gpu.ntt
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_ntt_fwd_stage0_w32_deg16
  func.func @test_ntt_modup(%out: memref<4x16xi32>, %in: memref<4x16xi32>) {
    cheddar_gpu.ntt %out, %in <{
      in_form = #cheddar_gpu.form<coeff>,
      out_form = #cheddar_gpu.form<eval>,
      montgomery_conversion = false,
      skip_range = array<i64: 1, 3>,
      out_limb_range = array<i64: 0, 4>
    }> : memref<4x16xi32>
    return
  }

  // CHECK: func.func @test_ntt_level_invariance
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_ntt_fwd_stage0_w32_deg16
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_ntt_fwd_stage0_w32_deg16
  func.func @test_ntt_level_invariance(%out1: memref<1x16xi32>, %in1: memref<1x16xi32>,
                                       %out2: memref<4x16xi32>, %in2: memref<4x16xi32>) {
    cheddar_gpu.ntt %out1, %in1 <{
      in_form = #cheddar_gpu.form<coeff>,
      out_form = #cheddar_gpu.form<eval>,
      montgomery_conversion = false
    }> : memref<1x16xi32>
    cheddar_gpu.ntt %out2, %in2 <{
      in_form = #cheddar_gpu.form<coeff>,
      out_form = #cheddar_gpu.form<eval>,
      montgomery_conversion = false
    }> : memref<4x16xi32>
    return
  }
}
