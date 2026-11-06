// RUN: heir-opt --verify-diagnostics --split-input-file %s

// -----
// Negative level for mod_up

func.func @test_mod_up_negative_level(%out0: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @below {{failed to verify that level is non-negative, or -1 when 'outs' has one more limb than 'in'}}
  cheddar_gpu.mod_up outs(%out0), %in <level = -1> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  return
}

// -----
// mod_up outs variadic types mismatch (CheddarGPU_AllVariadicTypesEqual)

func.func @test_mod_up_variadic_types_mismatch(
    %out0: memref<5x65536xi32>, %out1: memref<4x65536xi32>,
    %in: memref<4x65536xi32>) {
  // expected-error @below {{failed to verify that all of {outs} have the same type}}
  cheddar_gpu.mod_up outs(%out0, %out1), %in <level = 1> : (memref<5x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  return
}

// -----
// mod_down out limbs not strictly less than in limbs

func.func @test_mod_down_invalid_limbs(%out: memref<5x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @below {{failed to verify that 'out' has fewer limbs than 'in'}}
  cheddar_gpu.mod_down %out, %in <level = 1> : (memref<5x65536xi32>, memref<4x65536xi32>) -> ()
  return
}

// -----
// mod_down_epilogue pad_end out of bounds

func.func @test_mod_down_epilogue_pad_out_of_bounds(%out: memref<3x65536xi32>, %ntt_bconv: memref<3x65536xi32>, %inv_p_prod: memref<3xi32>) {
  // expected-error @below {{pad range [0, 5] out of bounds for output with 3 limbs}}
  cheddar_gpu.mod_down_epilogue %out, %ntt_bconv, %inv_p_prod <pad_end = 5, pad_start = 0> : (memref<3x65536xi32>, memref<3x65536xi32>, memref<3xi32>) -> ()
  return
}

// -----
// mod_down_epilogue src2 too small for pad range

func.func @test_mod_down_epilogue_src2_too_small(
    %out: memref<4x65536xi32>, %ntt_bconv: memref<4x65536xi32>,
    %inv_p_prod: memref<4xi32>, %src2: memref<2x65536xi32>) {
  // expected-error @below {{failed to verify that 'src2' (if present) covers the pad range}}
  cheddar_gpu.mod_down_epilogue %out, %ntt_bconv, %inv_p_prod, src2(%src2) <pad_start = 1, pad_end = 4> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4xi32>, memref<2x65536xi32>) -> ()
  return
}

// -----
// ntt_mod_down src2 too small for pad range + src2_offset

func.func @test_ntt_mod_down_src2_too_small(
    %out: memref<4x65536xi32>, %ntt_in: memref<4x65536xi32>,
    %inv_p_prod: memref<4xi32>, %src2: memref<3x65536xi32>) {
  // expected-error @below {{failed to verify that 'src2' (if present) covers the pad range}}
  cheddar_gpu.ntt_mod_down %out, %ntt_in, %inv_p_prod, src2(%src2) <pad_start = 1, pad_end = 3, src2_offset = 2> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4xi32>, memref<3x65536xi32>) -> ()
  return
}

// -----
// ntt_mod_down src2_padding too small for pad range

func.func @test_ntt_mod_down_src2_padding_too_small(
    %out: memref<4x65536xi32>, %ntt_in: memref<4x65536xi32>,
    %inv_p_prod: memref<4xi32>, %src2: memref<4x65536xi32>,
    %src2_padding: memref<2xi32>) {
  // expected-error @below {{failed to verify that 'src2_padding' (if present) covers the pad range}}
  cheddar_gpu.ntt_mod_down %out, %ntt_in, %inv_p_prod, src2(%src2), src2_padding(%src2_padding) <pad_start = 1, pad_end = 4> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4xi32>, memref<4x65536xi32>, memref<2xi32>) -> ()
  return
}

// -----
// ntt_mod_down prime_indices size mismatch with out limbs

func.func @test_ntt_mod_down_prime_indices_mismatch(
    %out: memref<4x65536xi32>, %ntt_in: memref<4x65536xi32>,
    %inv_p_prod: memref<4xi32>) {
  // expected-error @below {{failed to verify that 'prime_indices' (if set) has one entry per limb of 'out'}}
  cheddar_gpu.ntt_mod_down %out, %ntt_in, %inv_p_prod <{pad_start = 0, pad_end = 4, prime_indices = array<i64: 0, 1>}> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4xi32>) -> ()
  return
}
