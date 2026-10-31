// RUN: heir-opt --cheddar-gpu-expand-keyswitch %s | FileCheck %s

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
  // CHECK: @test_expand_ks_no_mod_down
  // CHECK-NOT: cheddar_gpu.key_switch_no_mod_down
  // CHECK: cheddar_gpu.mod_up
  // CHECK: cheddar_gpu.cp_accum_2
  // CHECK: cheddar_gpu.cp_accum_add
  // CHECK-NOT: cheddar_gpu.add
  func.func @test_expand_ks_no_mod_down(
      %out_bx: memref<5x16xi32>, %out_ax: memref<5x16xi32>,
      %in: memref<3x16xi32>, %in_bx: memref<3x16xi32>,
      %evk0_bx: memref<5x16xi32>, %evk1_bx: memref<5x16xi32>,
      %evk0_ax: memref<5x16xi32>, %evk1_ax: memref<5x16xi32>) {
    cheddar_gpu.key_switch_no_mod_down %out_bx, %out_ax, %in in_bx(%in_bx),
        evk_bx(%evk0_bx, %evk1_bx), evk_ax(%evk0_ax, %evk1_ax)
        <level = 1, in_form = eval, out_form = eval>
        : (memref<5x16xi32>, memref<5x16xi32>, memref<3x16xi32>, memref<3x16xi32>,
           memref<5x16xi32>, memref<5x16xi32>, memref<5x16xi32>, memref<5x16xi32>) -> ()
    return
  }

  // CHECK: @test_expand_key_switch
  // CHECK-NOT: cheddar_gpu.key_switch
  // CHECK: cheddar_gpu.mod_up
  // CHECK: cheddar_gpu.cp_accum_2
  // CHECK: cheddar_gpu.cp_accum_add
  // CHECK: cheddar_gpu.mod_down
  // CHECK: cheddar_gpu.mod_down
  func.func @test_expand_key_switch(
      %out_bx: memref<3x16xi32>, %out_ax: memref<3x16xi32>,
      %in_bx: memref<3x16xi32>, %in_ax: memref<3x16xi32>,
      %evk0_bx: memref<5x16xi32>, %evk1_bx: memref<5x16xi32>,
      %evk0_ax: memref<5x16xi32>, %evk1_ax: memref<5x16xi32>) {
    cheddar_gpu.key_switch %out_bx, %out_ax, %in_bx, %in_ax,
        evk_bx(%evk0_bx, %evk1_bx), evk_ax(%evk0_ax, %evk1_ax)
        <level = 1, in_form = eval, out_form = eval>
        : (memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>,
           memref<5x16xi32>, memref<5x16xi32>, memref<5x16xi32>, memref<5x16xi32>) -> ()
    return
  }

  // CHECK: @test_expand_relinearize
  // CHECK-NOT: cheddar_gpu.relinearize
  // CHECK: cheddar_gpu.mod_up
  // CHECK: cheddar_gpu.cp_accum_2
  // CHECK: cheddar_gpu.cp_accum_add_2
  // CHECK: cheddar_gpu.mod_down
  // CHECK: cheddar_gpu.mod_down
  func.func @test_expand_relinearize(
      %out_bx: memref<3x16xi32>, %out_ax: memref<3x16xi32>,
      %in_bx: memref<3x16xi32>, %in_ax: memref<3x16xi32>, %in_rx: memref<3x16xi32>,
      %evk0_bx: memref<5x16xi32>, %evk1_bx: memref<5x16xi32>,
      %evk0_ax: memref<5x16xi32>, %evk1_ax: memref<5x16xi32>) {
    cheddar_gpu.relinearize %out_bx, %out_ax, %in_bx, %in_ax, %in_rx,
        evk_bx(%evk0_bx, %evk1_bx), evk_ax(%evk0_ax, %evk1_ax)
        <level = 1, in_form = eval, out_form = eval>
        : (memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>,
           memref<5x16xi32>, memref<5x16xi32>, memref<5x16xi32>, memref<5x16xi32>) -> ()
    return
  }

  // CHECK: @test_expand_relinearize_rescale
  // CHECK-NOT: cheddar_gpu.relinearize_rescale
  // CHECK: cheddar_gpu.mod_up
  // CHECK: cheddar_gpu.cp_accum_2
  // CHECK: cheddar_gpu.cp_accum_add_2
  // CHECK: cheddar_gpu.mod_down_rescale
  // CHECK: cheddar_gpu.mod_down_rescale
  func.func @test_expand_relinearize_rescale(
      %out_bx: memref<2x16xi32>, %out_ax: memref<2x16xi32>,
      %in_bx: memref<3x16xi32>, %in_ax: memref<3x16xi32>, %in_rx: memref<3x16xi32>,
      %evk0_bx: memref<5x16xi32>, %evk1_bx: memref<5x16xi32>,
      %evk0_ax: memref<5x16xi32>, %evk1_ax: memref<5x16xi32>) {
    cheddar_gpu.relinearize_rescale %out_bx, %out_ax, %in_bx, %in_ax, %in_rx,
        evk_bx(%evk0_bx, %evk1_bx), evk_ax(%evk0_ax, %evk1_ax)
        <level = 1, in_form = eval, out_form = eval>
        : (memref<2x16xi32>, memref<2x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>,
           memref<5x16xi32>, memref<5x16xi32>, memref<5x16xi32>, memref<5x16xi32>) -> ()
    return
  }

  // CHECK: @test_expand_hrot
  // CHECK-NOT: cheddar_gpu.hrot
  // CHECK: cheddar_gpu.mod_up
  // CHECK: cheddar_gpu.cp_accum_2
  // CHECK: cheddar_gpu.cp_accum_add
  // CHECK: cheddar_gpu.mod_down
  // CHECK: cheddar_gpu.mod_down
  // CHECK: cheddar_gpu.permute_2
  func.func @test_expand_hrot(
      %out_bx: memref<3x16xi32>, %out_ax: memref<3x16xi32>,
      %in_bx: memref<3x16xi32>, %in_ax: memref<3x16xi32>,
      %evk0_bx: memref<5x16xi32>, %evk1_bx: memref<5x16xi32>,
      %evk0_ax: memref<5x16xi32>, %evk1_ax: memref<5x16xi32>) {
    cheddar_gpu.hrot %out_bx, %out_ax, %in_bx, %in_ax,
        evk_bx(%evk0_bx, %evk1_bx), evk_ax(%evk0_ax, %evk1_ax)
        <level = 1, rot_dist = 1, in_form = eval, out_form = eval>
        : (memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>,
           memref<5x16xi32>, memref<5x16xi32>, memref<5x16xi32>, memref<5x16xi32>) -> ()
    return
  }

  // CHECK: @test_expand_hconj
  // CHECK-NOT: cheddar_gpu.hconj
  // CHECK: cheddar_gpu.mod_up
  // CHECK: cheddar_gpu.cp_accum_2
  // CHECK: cheddar_gpu.cp_accum_add
  // CHECK: cheddar_gpu.mod_down
  // CHECK: cheddar_gpu.mod_down
  // CHECK: cheddar_gpu.permute_2
  func.func @test_expand_hconj(
      %out_bx: memref<3x16xi32>, %out_ax: memref<3x16xi32>,
      %in_bx: memref<3x16xi32>, %in_ax: memref<3x16xi32>,
      %evk0_bx: memref<5x16xi32>, %evk1_bx: memref<5x16xi32>,
      %evk0_ax: memref<5x16xi32>, %evk1_ax: memref<5x16xi32>) {
    cheddar_gpu.hconj %out_bx, %out_ax, %in_bx, %in_ax,
        evk_bx(%evk0_bx, %evk1_bx), evk_ax(%evk0_ax, %evk1_ax)
        <level = 1, in_form = eval, out_form = eval>
        : (memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>,
           memref<5x16xi32>, memref<5x16xi32>, memref<5x16xi32>, memref<5x16xi32>) -> ()
    return
  }

  // CHECK: @test_expand_hrot_add
  // CHECK-NOT: cheddar_gpu.hrot_add
  // CHECK: cheddar_gpu.mod_up
  // CHECK: cheddar_gpu.cp_accum_2
  // CHECK: cheddar_gpu.cp_accum_add
  // CHECK: cheddar_gpu.mod_down
  // CHECK: cheddar_gpu.mod_down
  // CHECK: cheddar_gpu.permute_accum_2
  // CHECK-NOT: cheddar_gpu.add
  func.func @test_expand_hrot_add(
      %out_bx: memref<3x16xi32>, %out_ax: memref<3x16xi32>,
      %in1_bx: memref<3x16xi32>, %in1_ax: memref<3x16xi32>,
      %in2_bx: memref<3x16xi32>, %in2_ax: memref<3x16xi32>,
      %evk0_bx: memref<5x16xi32>, %evk1_bx: memref<5x16xi32>,
      %evk0_ax: memref<5x16xi32>, %evk1_ax: memref<5x16xi32>) {
    cheddar_gpu.hrot_add %out_bx, %out_ax, %in1_bx, %in1_ax, %in2_bx, %in2_ax,
        evk_bx(%evk0_bx, %evk1_bx), evk_ax(%evk0_ax, %evk1_ax)
        <level = 1, rot_dist = 1, in_form = eval, out_form = eval>
        : (memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>,
           memref<3x16xi32>, memref<3x16xi32>,
           memref<5x16xi32>, memref<5x16xi32>, memref<5x16xi32>, memref<5x16xi32>) -> ()
    return
  }
}
