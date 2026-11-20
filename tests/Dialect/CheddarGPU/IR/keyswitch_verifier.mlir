// RUN: heir-opt --verify-diagnostics --split-input-file %s

// -----
// Negative level for key_switch (output limbs != 1)

func.func @test_key_switch_negative_level(
    %out_bx: memref<3x16xi32>, %out_ax: memref<3x16xi32>,
    %in_bx: memref<3x16xi32>, %in_ax: memref<3x16xi32>,
    %evk0_bx: memref<5x16xi32>, %evk0_ax: memref<5x16xi32>) {
  // expected-error @below {{failed to verify that level is non-negative, or -1 when 'out_bx' has 1 limb}}
  cheddar_gpu.key_switch %out_bx, %out_ax, %in_bx, %in_ax,
      evk_bx(%evk0_bx), evk_ax(%evk0_ax)
      <level = -1>
      : (memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>,
         memref<5x16xi32>, memref<5x16xi32>) -> ()
  return
}

// -----
// Negative level for key_switch_no_mod_down (output limbs != 2)

func.func @test_key_switch_no_mod_down_negative_level(
    %out_bx: memref<5x16xi32>, %out_ax: memref<5x16xi32>,
    %in: memref<3x16xi32>,
    %evk0_bx: memref<5x16xi32>, %evk0_ax: memref<5x16xi32>) {
  // expected-error @below {{failed to verify that level is non-negative, or -1 when 'out_bx' has 2 limbs}}
  cheddar_gpu.key_switch_no_mod_down %out_bx, %out_ax, %in,
      evk_bx(%evk0_bx), evk_ax(%evk0_ax)
      <level = -1>
      : (memref<5x16xi32>, memref<5x16xi32>, memref<3x16xi32>,
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
  // expected-error @below {{failed to verify that all of {evk_bx, evk_ax} have same count}}
  cheddar_gpu.relinearize %out_bx, %out_ax, %in_bx, %in_ax, %in_rx,
      evk_bx(%evk0_bx, %evk1_bx), evk_ax(%evk0_ax)
      <level = 1>
      : (memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>,
         memref<5x16xi32>, memref<5x16xi32>, memref<5x16xi32>) -> ()
  return
}

// -----
// key_switch_no_mod_down out limbs not strictly greater than in limbs

func.func @test_ks_no_mod_down_limb_count(
    %out_bx: memref<2x16xi32>, %out_ax: memref<2x16xi32>,
    %in: memref<3x16xi32>,
    %evk0_bx: memref<5x16xi32>, %evk0_ax: memref<5x16xi32>) {
  // expected-error @below {{failed to verify that 'out_bx' has more limbs than 'in'}}
  cheddar_gpu.key_switch_no_mod_down %out_bx, %out_ax, %in,
      evk_bx(%evk0_bx), evk_ax(%evk0_ax)
      <level = 1>
      : (memref<2x16xi32>, memref<2x16xi32>, memref<3x16xi32>,
         memref<5x16xi32>, memref<5x16xi32>) -> ()
  return
}

// -----
// relinearize_rescale out limbs not strictly less than in limbs

func.func @test_relinearize_rescale_invalid_limbs(
    %out_bx: memref<4x16xi32>, %out_ax: memref<4x16xi32>,
    %in_bx: memref<3x16xi32>, %in_ax: memref<3x16xi32>, %in_rx: memref<3x16xi32>,
    %evk0_bx: memref<5x16xi32>, %evk0_ax: memref<5x16xi32>) {
  // expected-error @below {{failed to verify that 'out_bx' has fewer limbs than 'in_bx'}}
  cheddar_gpu.relinearize_rescale %out_bx, %out_ax, %in_bx, %in_ax, %in_rx,
      evk_bx(%evk0_bx), evk_ax(%evk0_ax)
      <level = 1>
      : (memref<4x16xi32>, memref<4x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>,
         memref<5x16xi32>, memref<5x16xi32>) -> ()
  return
}

// -----
// relinearize evk degree mismatch

func.func @test_relinearize_evk_degree_mismatch(
    %out_bx: memref<3x16xi32>, %out_ax: memref<3x16xi32>,
    %in_bx: memref<3x16xi32>, %in_ax: memref<3x16xi32>, %in_rx: memref<3x16xi32>,
    %evk0_bx: memref<5x32xi32>, %evk0_ax: memref<5x32xi32>) {
  // expected-error @below {{failed to verify that every 'evk_bx' has the degree and word type of 'out_bx'}}
  cheddar_gpu.relinearize %out_bx, %out_ax, %in_bx, %in_ax, %in_rx,
      evk_bx(%evk0_bx), evk_ax(%evk0_ax)
      <level = 1>
      : (memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>,
         memref<5x32xi32>, memref<5x32xi32>) -> ()
  return
}

// -----
// key_switch_no_mod_down evk limb count less than out_bx limb count

func.func @test_key_switch_no_mod_down_evk_limbs(
    %out_bx: memref<5x16xi32>, %out_ax: memref<5x16xi32>,
    %in: memref<3x16xi32>,
    %evk0_bx: memref<3x16xi32>, %evk0_ax: memref<3x16xi32>) {
  // expected-error @below {{failed to verify that every 'evk_bx' has dim 0 >= dim 0 of 'out_bx'}}
  cheddar_gpu.key_switch_no_mod_down %out_bx, %out_ax, %in,
      evk_bx(%evk0_bx), evk_ax(%evk0_ax)
      <level = 1>
      : (memref<5x16xi32>, memref<5x16xi32>, memref<3x16xi32>,
         memref<3x16xi32>, memref<3x16xi32>) -> ()
  return
}

// -----
// key_switch_no_mod_down evk_ax limb count less than out_bx limb count

func.func @test_key_switch_no_mod_down_evk_ax_limbs(
    %out_bx: memref<5x16xi32>, %out_ax: memref<5x16xi32>,
    %in: memref<3x16xi32>,
    %evk0_bx: memref<5x16xi32>, %evk0_ax: memref<3x16xi32>) {
  // expected-error @below {{failed to verify that all of {evk_bx, evk_ax} have same type}}
  cheddar_gpu.key_switch_no_mod_down %out_bx, %out_ax, %in,
      evk_bx(%evk0_bx), evk_ax(%evk0_ax)
      <level = 1>
      : (memref<5x16xi32>, memref<5x16xi32>, memref<3x16xi32>,
         memref<5x16xi32>, memref<3x16xi32>) -> ()
  return
}

// -----
// key_switch evk limb count less than out_bx limb count

func.func @test_key_switch_evk_limbs(
    %out_bx: memref<3x16xi32>, %out_ax: memref<3x16xi32>,
    %in_bx: memref<3x16xi32>, %in_ax: memref<3x16xi32>,
    %evk0_bx: memref<2x16xi32>, %evk0_ax: memref<2x16xi32>) {
  // expected-error @below {{failed to verify that every 'evk_bx' has dim 0 >= dim 0 of 'out_bx'}}
  cheddar_gpu.key_switch %out_bx, %out_ax, %in_bx, %in_ax,
      evk_bx(%evk0_bx), evk_ax(%evk0_ax)
      <level = 1>
      : (memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>,
         memref<2x16xi32>, memref<2x16xi32>) -> ()
  return
}

// -----
// relinearize_rescale evk limb count covers out_bx (2) but not in_bx (3)

func.func @test_relinearize_rescale_evk_limbs(
    %out_bx: memref<2x16xi32>, %out_ax: memref<2x16xi32>,
    %in_bx: memref<3x16xi32>, %in_ax: memref<3x16xi32>, %in_rx: memref<3x16xi32>,
    %evk0_bx: memref<2x16xi32>, %evk0_ax: memref<2x16xi32>) {
  // expected-error @below {{failed to verify that every 'evk_bx' has dim 0 >= dim 0 of 'in_bx'}}
  cheddar_gpu.relinearize_rescale %out_bx, %out_ax, %in_bx, %in_ax, %in_rx,
      evk_bx(%evk0_bx), evk_ax(%evk0_ax)
      <level = 1>
      : (memref<2x16xi32>, memref<2x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>,
         memref<2x16xi32>, memref<2x16xi32>) -> ()
  return
}

// -----
// evk_bx digits have mismatched limb counts

func.func @test_key_switch_evk_bx_digit_mismatch(
    %out_bx: memref<3x16xi32>, %out_ax: memref<3x16xi32>,
    %in_bx: memref<3x16xi32>, %in_ax: memref<3x16xi32>,
    %evk0_bx: memref<5x16xi32>, %evk1_bx: memref<4x16xi32>,
    %evk0_ax: memref<5x16xi32>, %evk1_ax: memref<5x16xi32>) {
  // expected-error @below {{failed to verify that all elements of 'evk_bx' have same type}}
  cheddar_gpu.key_switch %out_bx, %out_ax, %in_bx, %in_ax,
      evk_bx(%evk0_bx, %evk1_bx), evk_ax(%evk0_ax, %evk1_ax)
      <level = 1>
      : (memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>, memref<3x16xi32>,
         memref<5x16xi32>, memref<4x16xi32>, memref<5x16xi32>, memref<5x16xi32>) -> ()
  return
}
