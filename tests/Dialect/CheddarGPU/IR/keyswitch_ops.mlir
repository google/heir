// RUN: heir-opt %s | heir-opt | FileCheck %s

// CHECK: func.func @test_key_switch_no_mod_down
// CHECK: cheddar_gpu.key_switch_no_mod_down %{{.*}}, %{{.*}}, %{{.*}}, evk_bx(%{{.*}}, %{{.*}}), evk_ax(%{{.*}}, %{{.*}}) <level = 1, in_form = eval, out_form = eval>
func.func @test_key_switch_no_mod_down(
    %out_bx: memref<5x16xi32>, %out_ax: memref<5x16xi32>,
    %in: memref<3x16xi32>,
    %evk0_bx: memref<5x16xi32>, %evk1_bx: memref<5x16xi32>,
    %evk0_ax: memref<5x16xi32>, %evk1_ax: memref<5x16xi32>) {
  cheddar_gpu.key_switch_no_mod_down %out_bx, %out_ax, %in,
      evk_bx(%evk0_bx, %evk1_bx), evk_ax(%evk0_ax, %evk1_ax)
      <level = 1, in_form = eval, out_form = eval>
      : (memref<5x16xi32>, memref<5x16xi32>, memref<3x16xi32>,
         memref<5x16xi32>, memref<5x16xi32>, memref<5x16xi32>, memref<5x16xi32>) -> ()
  return
}

// CHECK: func.func @test_key_switch_no_mod_down_with_inputs
// CHECK: cheddar_gpu.key_switch_no_mod_down %{{.*}}, %{{.*}}, %{{.*}} in_bx(%{{.*}}) in_ax(%{{.*}}), evk_bx(%{{.*}}, %{{.*}}), evk_ax(%{{.*}}, %{{.*}}) <level = 1, in_form = eval, out_form = eval>
func.func @test_key_switch_no_mod_down_with_inputs(
    %out_bx: memref<5x16xi32>, %out_ax: memref<5x16xi32>,
    %in: memref<3x16xi32>, %in_bx: memref<3x16xi32>, %in_ax: memref<3x16xi32>,
    %evk0_bx: memref<5x16xi32>, %evk1_bx: memref<5x16xi32>,
    %evk0_ax: memref<5x16xi32>, %evk1_ax: memref<5x16xi32>) {
  cheddar_gpu.key_switch_no_mod_down %out_bx, %out_ax, %in in_bx(%in_bx) in_ax(%in_ax),
      evk_bx(%evk0_bx, %evk1_bx), evk_ax(%evk0_ax, %evk1_ax)
      <level = 1, in_form = eval, out_form = eval>
      : (memref<5x16xi32>, memref<5x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>,
         memref<5x16xi32>, memref<5x16xi32>, memref<5x16xi32>, memref<5x16xi32>) -> ()
  return
}

// CHECK: func.func @test_key_switch
// CHECK: cheddar_gpu.key_switch %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, evk_bx(%{{.*}}, %{{.*}}), evk_ax(%{{.*}}, %{{.*}}) <level = 1, in_form = eval, out_form = eval>
func.func @test_key_switch(
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

// CHECK: func.func @test_relinearize
// CHECK: cheddar_gpu.relinearize %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, evk_bx(%{{.*}}, %{{.*}}), evk_ax(%{{.*}}, %{{.*}}) <level = 1, in_form = eval, out_form = eval>
func.func @test_relinearize(
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

// CHECK: func.func @test_relinearize_rescale
// CHECK: cheddar_gpu.relinearize_rescale %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, evk_bx(%{{.*}}, %{{.*}}), evk_ax(%{{.*}}, %{{.*}}) <level = 1, in_form = eval, out_form = eval>
func.func @test_relinearize_rescale(
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

// CHECK: func.func @test_hrot
// CHECK: cheddar_gpu.hrot %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, evk_bx(%{{.*}}, %{{.*}}), evk_ax(%{{.*}}, %{{.*}}) <level = 1, rot_dist = 1, galois = 5, in_form = eval, out_form = eval>
func.func @test_hrot(
    %out_bx: memref<3x16xi32>, %out_ax: memref<3x16xi32>,
    %in_bx: memref<3x16xi32>, %in_ax: memref<3x16xi32>,
    %evk0_bx: memref<5x16xi32>, %evk1_bx: memref<5x16xi32>,
    %evk0_ax: memref<5x16xi32>, %evk1_ax: memref<5x16xi32>) {
  cheddar_gpu.hrot %out_bx, %out_ax, %in_bx, %in_ax,
      evk_bx(%evk0_bx, %evk1_bx), evk_ax(%evk0_ax, %evk1_ax)
      <level = 1, rot_dist = 1, galois = 5, in_form = eval, out_form = eval>
      : (memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>,
         memref<5x16xi32>, memref<5x16xi32>, memref<5x16xi32>, memref<5x16xi32>) -> ()
  return
}

// CHECK: func.func @test_hconj
// CHECK: cheddar_gpu.hconj %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, evk_bx(%{{.*}}, %{{.*}}), evk_ax(%{{.*}}, %{{.*}}) <level = 1, in_form = eval, out_form = eval>
func.func @test_hconj(
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

// CHECK: func.func @test_hrot_add
// CHECK: cheddar_gpu.hrot_add %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, evk_bx(%{{.*}}, %{{.*}}), evk_ax(%{{.*}}, %{{.*}}) <level = 1, rot_dist = 1, in_form = eval, out_form = eval>
func.func @test_hrot_add(
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
