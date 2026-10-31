// RUN: heir-opt --verify-diagnostics --split-input-file %s

// -----
// Wrong input form for pseudo_mod_up (coeff instead of eval)

func.func @test_pseudo_mod_up_wrong_form(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @+1 {{pseudo_mod_up requires eval form for input and output}}
  cheddar_gpu.pseudo_mod_up %out, %in <in_form = coeff, out_form = eval, level = 2> : memref<4x65536xi32>
  return
}

// -----
// Negative level for mod_up

func.func @test_mod_up_negative_level(%out0: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @+1 {{level must be non-negative}}
  cheddar_gpu.mod_up (%out0), %in <in_form = eval, out_form = eval, level = -1> : (memref<4x65536xi32>), memref<4x65536xi32>
  return
}

// -----
// mod_down out limbs not strictly less than in limbs

func.func @test_mod_down_invalid_limbs(%out: memref<5x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @+1 {{output limb count (5) must be strictly less than input limb count (4)}}
  cheddar_gpu.mod_down %out, %in <in_form = eval, out_form = eval, level = 1> : memref<5x65536xi32>, memref<4x65536xi32>
  return
}

// -----
// rescale level 0 is invalid

func.func @test_rescale_level_zero(%out: memref<2x65536xi64>, %in: memref<3x65536xi64>) {
  // expected-error @+1 {{rescale is only valid for level > 0, got level 0}}
  cheddar_gpu.rescale %out, %in <in_form = eval, out_form = eval, level = 0> : memref<2x65536xi64>, memref<3x65536xi64>
  return
}

// -----
// mod_down_rescale degree mismatch

func.func @test_mod_down_rescale_degree_mismatch(%out: memref<2x32768xi64>, %in: memref<5x65536xi64>) {
  // expected-error @+1 {{polynomial degree mismatch between output (32768) and input (65536)}}
  cheddar_gpu.mod_down_rescale %out, %in <in_form = eval, out_form = eval, level = 2> : memref<2x32768xi64>, memref<5x65536xi64>
  return
}

// -----
// mod_down_epilogue pad_end out of bounds

func.func @test_mod_down_epilogue_pad_out_of_bounds(%out: memref<3x65536xi32>, %ntt_bconv: memref<3x65536xi32>, %inv_p_prod: memref<3xi32>) {
  // expected-error @+1 {{pad range [0, 5] out of bounds for output with 3 limbs}}
  cheddar_gpu.mod_down_epilogue %out, %ntt_bconv, %inv_p_prod <in_form = eval, out_form = eval, pad_end = 5, pad_start = 0> : memref<3x65536xi32>, memref<3x65536xi32>, memref<3xi32>
  return
}

// -----
// mod_down_epilogue inv_prime_prod wrong rank

func.func @test_mod_down_epilogue_inv_p_prod_rank(%out: memref<3x65536xi32>, %ntt_bconv: memref<3x65536xi32>, %inv_p_prod: memref<3x1xi32>) {
  // expected-error @+1 {{inv_prime_prod must be rank 1 (per-limb), but got rank 2}}
  cheddar_gpu.mod_down_epilogue %out, %ntt_bconv, %inv_p_prod <in_form = eval, out_form = eval, pad_end = 3, pad_start = 0> : memref<3x65536xi32>, memref<3x65536xi32>, memref<3x1xi32>
  return
}
