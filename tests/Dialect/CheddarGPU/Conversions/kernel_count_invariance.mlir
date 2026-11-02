// RUN: heir-opt %s --cheddar-gpu-to-gpu --gpu-kernel-outlining | FileCheck %s

// Verify that multiple elementwise operations on tensors with different numbers
// of limbs (levels 1, 2, 4) reuse the exact same memoized gpu.func kernel.

module attributes {
  cheddar_gpu.params = #cheddar_gpu.params<
    word = 32,
    log_n = 16,
    ter = [30539777],
    main = [974258177, 1081212929, 1196556289, 993263617],
    aux = [2147352577],
    level_config = [[[2 : i32, 1 : i32]]],
    alpha = 1,
    dnum = 0
  >
} {
  // CHECK: gpu.module @cheddar_kernels
  // CHECK-DAG: gpu.func @cheddar_gpu_add_w32_deg65536
  // CHECK-DAG: gpu.func @cheddar_gpu_copy_w32_deg65536
  // CHECK-DAG: gpu.func @cheddar_gpu_bconv_w32_deg65536
  // CHECK-DAG: gpu.func @cheddar_gpu_mod_down_epilogue_nosrc2_w32_deg65536

  // CHECK: func.func @test_level_1
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_add_w32_deg65536
  func.func @test_level_1(%out: memref<1x65536xi32>, %lhs: memref<1x65536xi32>, %rhs: memref<1x65536xi32>) {
    cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<1x65536xi32>
    return
  }

  // CHECK: func.func @test_level_2
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_add_w32_deg65536
  func.func @test_level_2(%out: memref<2x65536xi32>, %lhs: memref<2x65536xi32>, %rhs: memref<2x65536xi32>) {
    cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<2x65536xi32>
    return
  }

  // CHECK: func.func @test_level_4
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_add_w32_deg65536
  func.func @test_level_4(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
    cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>
    }> : memref<4x65536xi32>
    return
  }

  // CHECK: func.func @test_copy_level_1
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_copy_w32_deg65536
  func.func @test_copy_level_1(%out: memref<1x65536xi32>, %in: memref<1x65536xi32>) {
    cheddar_gpu.copy %out, %in : memref<1x65536xi32>, memref<1x65536xi32>
    return
  }

  // CHECK: func.func @test_copy_level_4
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_copy_w32_deg65536
  func.func @test_copy_level_4(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
    cheddar_gpu.copy %out, %in : memref<4x65536xi32>, memref<4x65536xi32>
    return
  }

  // CHECK: func.func @test_bconv_level_1
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_bconv_w32_deg65536
  func.func @test_bconv_level_1(%out: memref<1x65536xi32>, %in: memref<1x65536xi32>, %tbl: memref<1x1xi32>) {
    cheddar_gpu.bconv %out, %in, %tbl <{
      in_form = #cheddar_gpu.form<coeff_centered>,
      out_form = #cheddar_gpu.form<coeff>,
      in_limb_range = array<i64: 0, 1>,
      out_limb_range = array<i64: 0, 1>
    }> : memref<1x65536xi32>, memref<1x65536xi32>, memref<1x1xi32>
    return
  }

  // CHECK: func.func @test_bconv_level_4
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_bconv_w32_deg65536
  func.func @test_bconv_level_4(%out: memref<4x65536xi32>, %in: memref<2x65536xi32>, %tbl: memref<4x2xi32>) {
    cheddar_gpu.bconv %out, %in, %tbl <{
      in_form = #cheddar_gpu.form<coeff_centered>,
      out_form = #cheddar_gpu.form<coeff>,
      in_limb_range = array<i64: 0, 2>,
      out_limb_range = array<i64: 0, 4>
    }> : memref<4x65536xi32>, memref<2x65536xi32>, memref<4x2xi32>
    return
  }

  // CHECK: func.func @test_epilogue_level_1
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_mod_down_epilogue_nosrc2_w32_deg65536
  func.func @test_epilogue_level_1(%out: memref<1x65536xi32>, %ntt: memref<1x65536xi32>, %p: memref<1xi32>) {
    cheddar_gpu.mod_down_epilogue %out, %ntt, %p <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>,
      pad_start = 0 : i64,
      pad_end = 0 : i64
    }> : memref<1x65536xi32>, memref<1x65536xi32>, memref<1xi32>
    return
  }

  // CHECK: func.func @test_epilogue_level_4
  // CHECK: gpu.launch_func{{.*}}@cheddar_kernels::@cheddar_gpu_mod_down_epilogue_nosrc2_w32_deg65536
  func.func @test_epilogue_level_4(%out: memref<4x65536xi32>, %ntt: memref<4x65536xi32>, %p: memref<4xi32>) {
    cheddar_gpu.mod_down_epilogue %out, %ntt, %p <{
      in_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>,
      pad_start = 0 : i64,
      pad_end = 0 : i64
    }> : memref<4x65536xi32>, memref<4x65536xi32>, memref<4xi32>
    return
  }
}
