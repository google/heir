// RUN: heir-opt --verify-diagnostics --split-input-file %s

// Wrong form (coeff instead of eval) for key_switch_no_mod_down

func.func @test_ks_no_mod_down_wrong_form(
    %out_bx: memref<5x16xi32>, %out_ax: memref<5x16xi32>,
    %in: memref<3x16xi32>,
    %evk0_bx: memref<5x16xi32>, %evk0_ax: memref<5x16xi32>) {
  // expected-error @+1 {{key_switch_no_mod_down requires eval form for input and output}}
  cheddar_gpu.key_switch_no_mod_down %out_bx, %out_ax, %in,
      evk_bx(%evk0_bx), evk_ax(%evk0_ax)
      <level = 1, in_form = coeff, out_form = eval>
      : (memref<5x16xi32>, memref<5x16xi32>, memref<3x16xi32>,
         memref<5x16xi32>, memref<5x16xi32>) -> ()
  return
}

// -----
// Negative level for key_switch

func.func @test_key_switch_negative_level(
    %out_bx: memref<3x16xi32>, %out_ax: memref<3x16xi32>,
    %in_bx: memref<3x16xi32>, %in_ax: memref<3x16xi32>,
    %evk0_bx: memref<5x16xi32>, %evk0_ax: memref<5x16xi32>) {
  // expected-error @+1 {{level must be non-negative}}
  cheddar_gpu.key_switch %out_bx, %out_ax, %in_bx, %in_ax,
      evk_bx(%evk0_bx), evk_ax(%evk0_ax)
      <level = -1, in_form = eval, out_form = eval>
      : (memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>,
         memref<5x16xi32>, memref<5x16xi32>) -> ()
  return
}

// -----
// EVK digit count mismatch

func.func @test_relinearize_evk_mismatch(
    %out_bx: memref<3x16xi32>, %out_ax: memref<3x16xi32>,
    %in_bx: memref<3x16xi32>, %in_ax: memref<3x16xi32>, %in_rx: memref<3x16xi32>,
    %evk0_bx: memref<5x16xi32>, %evk1_bx: memref<5x16xi32>,
    %evk0_ax: memref<5x16xi32>) {
  // expected-error @+1 {{evaluation key digit counts must match between evk_bx (2) and evk_ax (1)}}
  cheddar_gpu.relinearize %out_bx, %out_ax, %in_bx, %in_ax, %in_rx,
      evk_bx(%evk0_bx, %evk1_bx), evk_ax(%evk0_ax)
      <level = 1, in_form = eval, out_form = eval>
      : (memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>,
         memref<5x16xi32>, memref<5x16xi32>, memref<5x16xi32>) -> ()
  return
}

// -----
// relinearize_rescale level 0 invalid

func.func @test_relinearize_rescale_level_zero(
    %out_bx: memref<2x16xi32>, %out_ax: memref<2x16xi32>,
    %in_bx: memref<3x16xi32>, %in_ax: memref<3x16xi32>, %in_rx: memref<3x16xi32>,
    %evk0_bx: memref<5x16xi32>, %evk0_ax: memref<5x16xi32>) {
  // expected-error @+1 {{relinearize_rescale is only valid for level > 0, got level 0}}
  cheddar_gpu.relinearize_rescale %out_bx, %out_ax, %in_bx, %in_ax, %in_rx,
      evk_bx(%evk0_bx), evk_ax(%evk0_ax)
      <level = 0, in_form = eval, out_form = eval>
      : (memref<2x16xi32>, memref<2x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>,
         memref<5x16xi32>, memref<5x16xi32>) -> ()
  return
}

// -----
// hrot degree mismatch between in and out

func.func @test_hrot_degree_mismatch(
    %out_bx: memref<3x16xi32>, %out_ax: memref<3x16xi32>,
    %in_bx: memref<3x32xi32>, %in_ax: memref<3x32xi32>,
    %evk0_bx: memref<5x16xi32>, %evk0_ax: memref<5x16xi32>) {
  // expected-error @+1 {{operand shape mismatch between out_bx and in_bx}}
  cheddar_gpu.hrot %out_bx, %out_ax, %in_bx, %in_ax,
      evk_bx(%evk0_bx), evk_ax(%evk0_ax)
      <level = 1, rot_dist = 1, in_form = eval, out_form = eval>
      : (memref<3x16xi32>, memref<3x16xi32>, memref<3x32xi32>, memref<3x32xi32>,
         memref<5x16xi32>, memref<5x16xi32>) -> ()
  return
}
