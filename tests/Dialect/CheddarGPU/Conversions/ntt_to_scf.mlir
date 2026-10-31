// RUN: heir-opt --cheddar-gpu-to-scf %s | FileCheck %s

module attributes {
  cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 4, ter = [30539777, 37224449, 33292289, 32899073], main = [], aux = [], level_config = [[[1 : i32, 1 : i32]]], alpha = 1, dnum = 0>
} {
  // CHECK: @test_ntt
  // CHECK-NOT: cheddar_gpu.ntt
  // CHECK-NOT: arith.remui
  // CHECK-NOT: arith.remsi
  // CHECK: scf.for
  // CHECK: mod_arith.mont_mul
  // CHECK: mod_arith.add
  // CHECK: mod_arith.sub
  func.func @test_ntt(%out: memref<4x16xi32>, %in: memref<4x16xi32>) {
    cheddar_gpu.ntt %out, %in <{
      in_form = #cheddar_gpu.form<coeff>,
      out_form = #cheddar_gpu.form<eval>,
      montgomery_conversion = false,
      out_limb_range = array<i64: 2, 2>
    }> : memref<4x16xi32>
    return
  }

  // CHECK: @test_intt_none
  // CHECK-NOT: cheddar_gpu.intt
  // CHECK-NOT: arith.remui
  // CHECK-NOT: arith.remsi
  // CHECK: scf.for
  // CHECK: mod_arith.add
  // CHECK: mod_arith.sub
  // CHECK: mod_arith.mont_mul
  // CHECK: mod_arith.lift
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

  // CHECK: @test_intt_mult_const
  // CHECK-NOT: cheddar_gpu.intt
  // CHECK: scf.for
  // CHECK: mod_arith.mont_mul
  // CHECK: mod_arith.lift
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

  // CHECK: @test_intt_mult_const_normalize
  // CHECK-NOT: cheddar_gpu.intt
  // CHECK: scf.for
  // CHECK: mod_arith.mont_mul
  // CHECK: mod_arith.lift
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
}
