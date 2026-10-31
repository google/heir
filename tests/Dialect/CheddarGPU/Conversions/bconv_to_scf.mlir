// RUN: heir-opt --cheddar-gpu-to-scf %s | FileCheck %s

module attributes {
  cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 4, ter = [1000000007], main = [1000000009, 1000000087], aux = [1000000021, 1000000033], level_config = [[[1 : i32, 1 : i32]]], alpha = 1, dnum = 0>
} {
  // CHECK: @test_bconv_basic
  // CHECK-NOT: cheddar_gpu.bconv
  // CHECK-NOT: arith.remui
  // CHECK-NOT: arith.remsi
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: arith.extsi
  // CHECK: arith.muli
  // CHECK: arith.addi
  // CHECK: arith.select
  // CHECK: mod_arith.mont_reduce
  // CHECK: mod_arith.lift
  // CHECK: memref.store
  func.func @test_bconv_basic(%out: memref<5x16xi32>, %in: memref<2x16xi32>, %table: memref<2x2xi32>) {
    cheddar_gpu.bconv %out, %in, %table <{
      in_form = #cheddar_gpu.form<coeff_centered>,
      out_form = #cheddar_gpu.form<coeff>,
      out_limb_range = array<i64: 3, 2>
    }> : memref<5x16xi32>, memref<2x16xi32>, memref<2x2xi32>
    return
  }

  // CHECK: @test_bconv_skip
  // CHECK-NOT: cheddar_gpu.bconv
  // CHECK-NOT: arith.remui
  // CHECK-NOT: arith.remsi
  // CHECK: scf.for
  // CHECK: mod_arith.mont_reduce
  func.func @test_bconv_skip(%out: memref<5x16xi32>, %in: memref<2x16xi32>, %table: memref<2x2xi32>) {
    cheddar_gpu.bconv %out, %in, %table <{
      in_form = #cheddar_gpu.form<coeff_centered>,
      out_form = #cheddar_gpu.form<coeff>,
      skip_range = array<i64: 1, 3>,
      out_limb_range = array<i64: 1, 4>
    }> : memref<5x16xi32>, memref<2x16xi32>, memref<2x2xi32>
    return
  }
}
