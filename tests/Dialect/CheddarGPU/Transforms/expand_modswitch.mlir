// RUN: heir-opt --cheddar-gpu-expand-modswitch %s | FileCheck %s

module attributes {
  cheddar_gpu.params = #cheddar_gpu.params<
    word = 32,
    log_n = 4,
    ter = [1000000007, 1000000003],
    main = [1000000009, 1000000087, 1000000093],
    aux = [1000000021, 1000000033],
    level_config = [[[1 : i32, 1 : i32]], [[2 : i32, 1 : i32]]],
    alpha = 2,
    dnum = 0
  >
} {
  // CHECK: @test_pseudo_mod_up
  // CHECK-NOT: cheddar_gpu.pseudo_mod_up
  // CHECK: memref.get_global @__cheddar_pseudo_mod_up_p_prod_l1
  // CHECK: cheddar_gpu.mul_const
  func.func @test_pseudo_mod_up(%out: memref<3x16xi32>, %in: memref<3x16xi32>) {
    cheddar_gpu.pseudo_mod_up %out, %in <level = 1, in_form = eval, out_form = eval> : memref<3x16xi32>
    return
  }

  // CHECK: @test_mod_up
  // CHECK-NOT: cheddar_gpu.mod_up
  // CHECK: cheddar_gpu.intt
  // CHECK-NOT: cheddar_gpu.copy
  // CHECK: cheddar_gpu.bconv
  // CHECK: cheddar_gpu.ntt %{{.*}}, %{{.*}}, skip_copy_src(%{{.*}})
  func.func @test_mod_up(%out0: memref<5x16xi32>, %out1: memref<5x16xi32>, %in: memref<3x16xi32>) {
    cheddar_gpu.mod_up (%out0, %out1), %in <level = 1, in_form = eval, out_form = eval> : (memref<5x16xi32>, memref<5x16xi32>), memref<3x16xi32>
    return
  }

  // CHECK: @test_mod_down
  // CHECK-NOT: cheddar_gpu.mod_down
  // CHECK: cheddar_gpu.intt
  // CHECK: cheddar_gpu.bconv
  // CHECK-NOT: cheddar_gpu.mod_down_epilogue
  // CHECK: cheddar_gpu.ntt_mod_down
  func.func @test_mod_down(%out: memref<3x16xi32>, %in: memref<5x16xi32>) {
    cheddar_gpu.mod_down %out, %in <level = 1, in_form = eval, out_form = eval> : memref<3x16xi32>, memref<5x16xi32>
    return
  }

  // CHECK: @test_rescale
  // CHECK-NOT: cheddar_gpu.rescale
  // CHECK: cheddar_gpu.intt
  // CHECK: cheddar_gpu.bconv
  // CHECK-NOT: cheddar_gpu.mod_down_epilogue
  // CHECK: cheddar_gpu.ntt_mod_down
  func.func @test_rescale(%out: memref<2x16xi32>, %in: memref<3x16xi32>) {
    cheddar_gpu.rescale %out, %in <level = 1, in_form = eval, out_form = eval> : memref<2x16xi32>, memref<3x16xi32>
    return
  }

  // CHECK: @test_mod_down_rescale
  // CHECK-NOT: cheddar_gpu.mod_down_rescale
  // CHECK: cheddar_gpu.intt
  // CHECK-NOT: cheddar_gpu.intt
  // CHECK: cheddar_gpu.bconv
  // CHECK-NOT: cheddar_gpu.mod_down_epilogue
  // CHECK: cheddar_gpu.ntt_mod_down
  func.func @test_mod_down_rescale(%out: memref<2x16xi32>, %in: memref<5x16xi32>) {
    cheddar_gpu.mod_down_rescale %out, %in <level = 1, in_form = eval, out_form = eval> : memref<2x16xi32>, memref<5x16xi32>
    return
  }
}
