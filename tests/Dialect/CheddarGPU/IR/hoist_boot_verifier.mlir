// RUN: heir-opt --verify-diagnostics --split-input-file %s

// mod_up_to_max in dim 0 < 1 (CheddarGPU_DimAtLeast)

func.func @test_mod_up_to_max_in_dim_0(%out: memref<4x65536xi32>, %in: memref<0x65536xi32>) {
  // expected-error @below {{failed to verify that dim 0 of 'in' is >= 1}}
  cheddar_gpu.mod_up_to_max %out, %in : (memref<4x65536xi32>, memref<0x65536xi32>) -> ()
  return
}

// -----
// mod_up_to_max out_limb_range out of bounds

func.func @test_mod_up_to_max_range_oob(%out: memref<4x65536xi32>, %in: memref<1x65536xi32>) {
  // expected-error @below {{limb range [0, 10] out of bounds for memref with 4 limbs}}
  cheddar_gpu.mod_up_to_max %out, %in <{
    out_limb_range = array<i64: 0, 10>
  }> : (memref<4x65536xi32>, memref<1x65536xi32>) -> ()
  return
}

// -----
// zero_pad start > end

func.func @test_zero_pad_invalid_span(%out: memref<4x65536xi32>) {
  // expected-error @below {{failed to satisfy constraint: i64 dense array attribute with exactly 2 elements of the form [start >= 0, end >= start]}}
  cheddar_gpu.zero_pad %out <{limb_range = array<i64: 3, 1>}> : (memref<4x65536xi32>) -> ()
  return
}

// -----
// zero_pad end > numLimbs (CheddarGPU_SpanEndWithinDim0)

func.func @test_zero_pad_end_oob(%out: memref<4x65536xi32>) {
  // expected-error @below {{failed to verify that end of 'limb_range' is within dim 0 of 'out'}}
  cheddar_gpu.zero_pad %out <{limb_range = array<i64: 0, 10>}> : (memref<4x65536xi32>) -> ()
  return
}

// -----
// hoist_bs_fused count mismatch between out_bx and out_ax

func.func @test_hoist_bs_fused_count_mismatch(
    %out_bx: memref<3x65536xi32>, %out_ax1: memref<3x65536xi32>, %out_ax2: memref<3x65536xi32>,
    %mod_up: memref<3x65536xi32>, %evk_slab: memref<1024xi8>,
    %in_bx_pseudo_modup: memref<2x65536xi32>) {
  // expected-error @below {{failed to verify that all of {out_bx, out_ax, galois_factors, evk_num_q} have same count}}
  cheddar_gpu.hoist_bs_fused out_bx(%out_bx), out_ax(%out_ax1, %out_ax2), mod_up(%mod_up), %evk_slab, %in_bx_pseudo_modup <{
    galois_factors = array<i64: 5>,
    evk_byte_offsets = array<i64: 0>,
    evk_num_q = array<i64: 2>,
    num_q = 2 : i64,
    num_aux = 1 : i64,
    prime_offset = 0 : i64
  }> : (memref<3x65536xi32>, memref<3x65536xi32>, memref<3x65536xi32>, memref<3x65536xi32>, memref<1024xi8>, memref<2x65536xi32>) -> ()
  return
}

// -----
// hoist_bs_fused out_bx count not in [1, 8] (CheddarGPU_CountInRange)

func.func @test_hoist_bs_fused_count_in_range(
    %mod_up: memref<3x65536xi32>, %evk_slab: memref<1024xi8>,
    %in_bx_pseudo_modup: memref<2x65536xi32>) {
  // expected-error @below {{failed to verify that 'out_bx' has 1 to 8 entries}}
  cheddar_gpu.hoist_bs_fused out_bx(), out_ax(), mod_up(%mod_up), %evk_slab, %in_bx_pseudo_modup <{
    galois_factors = array<i64>,
    evk_byte_offsets = array<i64>,
    evk_num_q = array<i64>,
    num_q = 2 : i64,
    num_aux = 1 : i64,
    prime_offset = 0 : i64
  }> : (memref<3x65536xi32>, memref<1024xi8>, memref<2x65536xi32>) -> ()
  return
}

// -----
// hoist_bs_fused output limb count mismatch with num_q + num_aux (CheddarGPU_LimbCountEqualsNumQPlusNumAux)

func.func @test_hoist_bs_fused_limbs_mismatch(
    %out_bx: memref<3x65536xi32>, %out_ax: memref<3x65536xi32>,
    %mod_up: memref<3x65536xi32>, %evk_slab: memref<1024xi8>,
    %in_bx_pseudo_modup: memref<2x65536xi32>) {
  // expected-error @below {{failed to verify that dim 0 of 'out_bx' is num_q + num_aux}}
  cheddar_gpu.hoist_bs_fused out_bx(%out_bx), out_ax(%out_ax), mod_up(%mod_up), %evk_slab, %in_bx_pseudo_modup <{
    galois_factors = array<i64: 5>,
    evk_byte_offsets = array<i64: 0>,
    evk_num_q = array<i64: 2>,
    num_q = 2 : i64,
    num_aux = 2 : i64,
    prime_offset = 0 : i64
  }> : (memref<3x65536xi32>, memref<3x65536xi32>, memref<3x65536xi32>, memref<1024xi8>, memref<2x65536xi32>) -> ()
  return
}

// -----
// hoist_bs_fused in_bx_pseudo_modup dim 0 < num_q

func.func @test_hoist_bs_fused_pseudo_less_than_q(
    %out_bx: memref<3x65536xi32>, %out_ax: memref<3x65536xi32>,
    %mod_up: memref<3x65536xi32>, %evk_slab: memref<1024xi8>,
    %in_bx_pseudo_modup: memref<1x65536xi32>) {
  // expected-error @below {{failed to verify that dim 0 of 'in_bx_pseudo_modup' is >= num_q}}
  cheddar_gpu.hoist_bs_fused out_bx(%out_bx), out_ax(%out_ax), mod_up(%mod_up), %evk_slab, %in_bx_pseudo_modup <{
    galois_factors = array<i64: 5>,
    evk_byte_offsets = array<i64: 0>,
    evk_num_q = array<i64: 2>,
    num_q = 2 : i64,
    num_aux = 1 : i64,
    prime_offset = 0 : i64
  }> : (memref<3x65536xi32>, memref<3x65536xi32>, memref<3x65536xi32>, memref<1024xi8>, memref<1x65536xi32>) -> ()
  return
}

// -----
// hoist_bs_fused evk_num_q < num_q

func.func @test_hoist_bs_fused_evk_num_q(
    %out_bx: memref<3x65536xi32>, %out_ax: memref<3x65536xi32>,
    %mod_up: memref<3x65536xi32>, %evk_slab: memref<1024xi8>,
    %in_bx_pseudo_modup: memref<2x65536xi32>) {
  // expected-error @below {{failed to verify that all elements of 'evk_num_q' are >= num_q}}
  cheddar_gpu.hoist_bs_fused out_bx(%out_bx), out_ax(%out_ax), mod_up(%mod_up), %evk_slab, %in_bx_pseudo_modup <{
    galois_factors = array<i64: 5>,
    evk_byte_offsets = array<i64: 0>,
    evk_num_q = array<i64: 1>,
    num_q = 2 : i64,
    num_aux = 1 : i64,
    prime_offset = 0 : i64
  }> : (memref<3x65536xi32>, memref<3x65536xi32>, memref<3x65536xi32>, memref<1024xi8>, memref<2x65536xi32>) -> ()
  return
}

// -----
// hoist_bs_fused evk_byte_offsets length mismatch

func.func @test_hoist_bs_fused_offsets_len(
    %out_bx: memref<3x65536xi32>, %out_ax: memref<3x65536xi32>,
    %mod_up: memref<3x65536xi32>, %evk_slab: memref<1024xi8>,
    %in_bx_pseudo_modup: memref<2x65536xi32>) {
  // expected-error @below {{evk_byte_offsets length (3) must be either R (1) or R*beta*2 (2)}}
  cheddar_gpu.hoist_bs_fused out_bx(%out_bx), out_ax(%out_ax), mod_up(%mod_up), %evk_slab, %in_bx_pseudo_modup <{
    galois_factors = array<i64: 5>,
    evk_byte_offsets = array<i64: 0, 0, 0>,
    evk_num_q = array<i64: 2>,
    num_q = 2 : i64,
    num_aux = 1 : i64,
    prime_offset = 0 : i64
  }> : (memref<3x65536xi32>, memref<3x65536xi32>, memref<3x65536xi32>, memref<1024xi8>, memref<2x65536xi32>) -> ()
  return
}

// -----
// hoist_bs_fused evk_byte_offsets not aligned to wordBytes

func.func @test_hoist_bs_fused_offset_align(
    %out_bx: memref<3x65536xi32>, %out_ax: memref<3x65536xi32>,
    %mod_up: memref<3x65536xi32>, %evk_slab: memref<1024xi8>,
    %in_bx_pseudo_modup: memref<2x65536xi32>) {
  // expected-error @below {{evk_byte_offsets[0] (3) must be non-negative and a multiple of 4}}
  cheddar_gpu.hoist_bs_fused out_bx(%out_bx), out_ax(%out_ax), mod_up(%mod_up), %evk_slab, %in_bx_pseudo_modup <{
    galois_factors = array<i64: 5>,
    evk_byte_offsets = array<i64: 3>,
    evk_num_q = array<i64: 2>,
    num_q = 2 : i64,
    num_aux = 1 : i64,
    prime_offset = 0 : i64
  }> : (memref<3x65536xi32>, memref<3x65536xi32>, memref<3x65536xi32>, memref<1024xi8>, memref<2x65536xi32>) -> ()
  return
}

// -----
// hoist_bs_fused evk_byte_offsets exceeds slab size

func.func @test_hoist_bs_fused_offset_bound(
    %out_bx: memref<3x65536xi32>, %out_ax: memref<3x65536xi32>,
    %mod_up: memref<3x65536xi32>, %evk_slab: memref<1024xi8>,
    %in_bx_pseudo_modup: memref<2x65536xi32>) {
  // expected-error @below {{evk_byte_offsets[0] (2048) exceeds evk_slab size (1024)}}
  cheddar_gpu.hoist_bs_fused out_bx(%out_bx), out_ax(%out_ax), mod_up(%mod_up), %evk_slab, %in_bx_pseudo_modup <{
    galois_factors = array<i64: 5>,
    evk_byte_offsets = array<i64: 2048>,
    evk_num_q = array<i64: 2>,
    num_q = 2 : i64,
    num_aux = 1 : i64,
    prime_offset = 0 : i64
  }> : (memref<3x65536xi32>, memref<3x65536xi32>, memref<3x65536xi32>, memref<1024xi8>, memref<2x65536xi32>) -> ()
  return
}

// -----
// hoist_gs_fused ciphertext limb count mismatch with num_q + num_aux

func.func @test_hoist_gs_fused_q_aux_mismatch(
    %out_bx: memref<3x65536xi32>, %out_ax: memref<3x65536xi32>,
    %in_bx: memref<3x65536xi32>, %in_ax: memref<3x65536xi32>,
    %consts_slab: memref<1024xi8>) {
  // expected-error @below {{failed to verify that dim 0 of 'out_bx' is num_q + num_aux}}
  cheddar_gpu.hoist_gs_fused out_bx(%out_bx), out_ax(%out_ax), in_bx(%in_bx), in_ax(%in_ax), %consts_slab <{
    pt_word_offsets = array<i64: 0>,
    num_q = 2 : i64,
    num_aux = 2 : i64,
    prime_offset = 0 : i64,
    pt_limbs = 4 : i64
  }> : (memref<3x65536xi32>, memref<3x65536xi32>, memref<3x65536xi32>, memref<3x65536xi32>, memref<1024xi8>) -> ()
  return
}

// -----
// hoist_gs_fused pt_limbs < ciphertext limbs

func.func @test_hoist_gs_fused_pt_limbs(
    %out_bx: memref<3x65536xi32>, %out_ax: memref<3x65536xi32>,
    %in_bx: memref<3x65536xi32>, %in_ax: memref<3x65536xi32>,
    %consts_slab: memref<1024xi8>) {
  // expected-error @below {{pt_limbs (2) must be >= ciphertext limbs (3)}}
  cheddar_gpu.hoist_gs_fused out_bx(%out_bx), out_ax(%out_ax), in_bx(%in_bx), in_ax(%in_ax), %consts_slab <{
    pt_word_offsets = array<i64: 0>,
    num_q = 2 : i64,
    num_aux = 1 : i64,
    prime_offset = 0 : i64,
    pt_limbs = 2 : i64
  }> : (memref<3x65536xi32>, memref<3x65536xi32>, memref<3x65536xi32>, memref<3x65536xi32>, memref<1024xi8>) -> ()
  return
}

// -----
// hoist_gs_fused pt_word_offsets length mismatch (G * B)

func.func @test_hoist_gs_fused_pt_offsets_len(
    %out_bx: memref<3x65536xi32>, %out_ax: memref<3x65536xi32>,
    %in_bx: memref<3x65536xi32>, %in_ax: memref<3x65536xi32>,
    %consts_slab: memref<1024xi8>) {
  // expected-error @below {{pt_word_offsets length (2) must equal G * B (1)}}
  cheddar_gpu.hoist_gs_fused out_bx(%out_bx), out_ax(%out_ax), in_bx(%in_bx), in_ax(%in_ax), %consts_slab <{
    pt_word_offsets = array<i64: 0, 1>,
    num_q = 2 : i64,
    num_aux = 1 : i64,
    prime_offset = 0 : i64,
    pt_limbs = 3 : i64
  }> : (memref<3x65536xi32>, memref<3x65536xi32>, memref<3x65536xi32>, memref<3x65536xi32>, memref<1024xi8>) -> ()
  return
}

// -----
// hoist_gs_fused pt_word_offsets elements must be >= -1

func.func @test_hoist_gs_fused_pt_offsets_ge_neg1(
    %out_bx: memref<3x65536xi32>, %out_ax: memref<3x65536xi32>,
    %in_bx: memref<3x65536xi32>, %in_ax: memref<3x65536xi32>,
    %consts_slab: memref<1024xi8>) {
  // expected-error @below {{failed to satisfy constraint: i64 dense array attribute where all elements are >= -1}}
  cheddar_gpu.hoist_gs_fused out_bx(%out_bx), out_ax(%out_ax), in_bx(%in_bx), in_ax(%in_ax), %consts_slab <{
    pt_word_offsets = array<i64: -2>,
    num_q = 2 : i64,
    num_aux = 1 : i64,
    prime_offset = 0 : i64,
    pt_limbs = 3 : i64
  }> : (memref<3x65536xi32>, memref<3x65536xi32>, memref<3x65536xi32>, memref<3x65536xi32>, memref<1024xi8>) -> ()
  return
}

// -----
// eval_mod offset not a multiple of 8

func.func @test_eval_mod_offset_not_multiple_of_8(
    %out_bx: memref<14x65536xi32>, %out_ax: memref<14x65536xi32>,
    %in_bx: memref<22x65536xi32>, %in_ax: memref<22x65536xi32>,
    %evk_slab: memref<1024xi8>) {
  // expected-error @below {{whose value is a multiple of 8}}
  cheddar_gpu.eval_mod ins(%in_bx, %in_ax, %evk_slab : memref<22x65536xi32>, memref<22x65536xi32>, memref<1024xi8>) outs(%out_bx, %out_ax : memref<14x65536xi32>, memref<14x65536xi32>) <{
    relin_evk_byte_offset = 5 : i64,
    relin_evk_num_q = 22 : i64,
    relin_evk_beta = 1 : i64
  }>
  return
}

// -----
// hoist_phase invalid phase for cts

func.func @test_hoist_phase_invalid_phase_cts(
    %out_bx: memref<25x65536xi32>, %out_ax: memref<25x65536xi32>,
    %in_bx: memref<26x65536xi32>, %in_ax: memref<26x65536xi32>,
    %consts_slab: memref<1024xi8>, %evk_slab: memref<1024xi8>) {
  // expected-error @below {{phase for 'cts' must be in [0, 3], but got 5}}
  cheddar_gpu.hoist_phase ins(%in_bx, %in_ax, %consts_slab, %evk_slab : memref<26x65536xi32>, memref<26x65536xi32>, memref<1024xi8>, memref<1024xi8>) outs(%out_bx, %out_ax : memref<25x65536xi32>, memref<25x65536xi32>) <{
    kind = #cheddar_gpu.hoist_phase_kind<cts>,
    phase = 5 : i64,
    num_slots = 256 : i64
  }>
  return
}

// -----
// hoist_phase invalid phase for stc

func.func @test_hoist_phase_invalid_phase_stc(
    %out_bx: memref<13x65536xi32>, %out_ax: memref<13x65536xi32>,
    %in_bx: memref<14x65536xi32>, %in_ax: memref<14x65536xi32>,
    %consts_slab: memref<1024xi8>, %evk_slab: memref<1024xi8>) {
  // expected-error @+1 {{phase for 'stc' must be in [0, 2], but got 3}}
  cheddar_gpu.hoist_phase ins(%in_bx, %in_ax, %consts_slab, %evk_slab : memref<14x65536xi32>, memref<14x65536xi32>, memref<1024xi8>, memref<1024xi8>) outs(%out_bx, %out_ax : memref<13x65536xi32>, memref<13x65536xi32>) <{
    kind = #cheddar_gpu.hoist_phase_kind<stc>,
    phase = 3 : i64,
    num_slots = 256 : i64
  }>
  return
}

// -----
// hoist_phase input limb count mismatch for cts

func.func @test_hoist_phase_in_limbs(
    %out_bx: memref<25x65536xi32>, %out_ax: memref<25x65536xi32>,
    %in_bx: memref<25x65536xi32>, %in_ax: memref<25x65536xi32>,
    %consts_slab: memref<1024xi8>, %evk_slab: memref<1024xi8>) {
  // expected-error @+1 {{input limb count for cts phase 0 must be 26, but got 25}}
  cheddar_gpu.hoist_phase ins(%in_bx, %in_ax, %consts_slab, %evk_slab : memref<25x65536xi32>, memref<25x65536xi32>, memref<1024xi8>, memref<1024xi8>) outs(%out_bx, %out_ax : memref<25x65536xi32>, memref<25x65536xi32>) <{
    kind = #cheddar_gpu.hoist_phase_kind<cts>,
    phase = 0 : i64,
    num_slots = 256 : i64
  }>
  return
}

// -----
// hoist_phase output limb count mismatch for stc

func.func @test_hoist_phase_out_limbs_stc(
    %out_bx: memref<12x65536xi32>, %out_ax: memref<12x65536xi32>,
    %in_bx: memref<14x65536xi32>, %in_ax: memref<14x65536xi32>,
    %consts_slab: memref<1024xi8>, %evk_slab: memref<1024xi8>) {
  // expected-error @+1 {{output limb count for stc phase 0 must be 13, but got 12}}
  cheddar_gpu.hoist_phase ins(%in_bx, %in_ax, %consts_slab, %evk_slab : memref<14x65536xi32>, memref<14x65536xi32>, memref<1024xi8>, memref<1024xi8>) outs(%out_bx, %out_ax : memref<12x65536xi32>, memref<12x65536xi32>) <{
    kind = #cheddar_gpu.hoist_phase_kind<stc>,
    phase = 0 : i64,
    num_slots = 256 : i64
  }>
  return
}

// -----
// boot out_bx dim 0 not in [1, 11] (CheddarGPU_DimInRange)

func.func @test_boot_dim_in_range(
    %out_bx: memref<12x65536xi64>, %out_ax: memref<12x65536xi64>,
    %in_bx: memref<14x65536xi64>, %in_ax: memref<14x65536xi64>,
    %consts_slab: memref<1024xi8>, %evk_slab: memref<1024xi8>) {
  // expected-error @+1 {{failed to verify that dim 0 of 'out_bx' is in [1, 11]}}
  cheddar_gpu.boot ins(%in_bx, %in_ax, %consts_slab, %evk_slab : memref<14x65536xi64>, memref<14x65536xi64>, memref<1024xi8>, memref<1024xi8>) outs(%out_bx, %out_ax : memref<12x65536xi64>, memref<12x65536xi64>) <{
    num_slots = 256 : i64
  }>
  return
}

// -----
// boot num_slots exceeds degree / 2 (CheddarGPU_NumSlotsFitDegree)

func.func @test_boot_num_slots_exceeds_degree(
    %out_bx: memref<11x512xi64>, %out_ax: memref<11x512xi64>,
    %in_bx: memref<14x512xi64>, %in_ax: memref<14x512xi64>,
    %consts_slab: memref<1024xi8>, %evk_slab: memref<1024xi8>) {
  // expected-error @+1 {{failed to verify that num_slots <= degree / 2}}
  cheddar_gpu.boot ins(%in_bx, %in_ax, %consts_slab, %evk_slab : memref<14x512xi64>, memref<14x512xi64>, memref<1024xi8>, memref<1024xi8>) outs(%out_bx, %out_ax : memref<11x512xi64>, memref<11x512xi64>) <{
    num_slots = 512 : i64
  }>
  return
}

// -----
// hoist_gs_fused pt_word_offsets out of consts_slab bounds

func.func @test_hoist_gs_fused_pt_offset_oob(
    %out_bx: memref<4x16xi32>, %out_ax: memref<4x16xi32>,
    %in_bx: memref<4x16xi32>, %in_ax: memref<4x16xi32>,
    %consts_slab: memref<1024xi8>) {
  // expected-error @+1 {{pt_word_offsets[0] (256) exceeds consts_slab size (1024 bytes)}}
  cheddar_gpu.hoist_gs_fused out_bx(%out_bx), out_ax(%out_ax), in_bx(%in_bx), in_ax(%in_ax), %consts_slab <{
    num_q = 4 : i64,
    num_aux = 0 : i64,
    prime_offset = 0 : i64,
    pt_limbs = 4 : i64,
    pt_word_offsets = array<i64: 256>
  }> : (memref<4x16xi32>, memref<4x16xi32>, memref<4x16xi32>, memref<4x16xi32>, memref<1024xi8>) -> ()
  return
}

// -----
// eval_mod relin_evk_byte_offset out of evk_slab bounds

func.func @test_eval_mod_evk_offset_oob(
    %out_bx: memref<14x65536xi32>, %out_ax: memref<14x65536xi32>,
    %in_bx: memref<22x65536xi32>, %in_ax: memref<22x65536xi32>,
    %evk_slab: memref<1024xi8>) {
  // expected-error @+1 {{failed to verify that 'relin_evk_byte_offset' is within 'evk_slab'}}
  cheddar_gpu.eval_mod ins(%in_bx, %in_ax, %evk_slab : memref<22x65536xi32>, memref<22x65536xi32>, memref<1024xi8>) outs(%out_bx, %out_ax : memref<14x65536xi32>, memref<14x65536xi32>) <{
    relin_evk_byte_offset = 1024 : i64,
    relin_evk_num_q = 22 : i64,
    relin_evk_beta = 1 : i64
  }>
  return
}

// -----
// hoist_gs_fused num_q + num_aux signed overflow safety

func.func @test_hoist_gs_fused_num_q_overflow(
    %out_bx: memref<4x16xi32>, %out_ax: memref<4x16xi32>,
    %in_bx: memref<4x16xi32>, %in_ax: memref<4x16xi32>,
    %consts_slab: memref<1024xi8>) {
  // expected-error @+1 {{failed to verify that dim 0 of 'out_bx' is num_q + num_aux}}
  cheddar_gpu.hoist_gs_fused out_bx(%out_bx), out_ax(%out_ax), in_bx(%in_bx), in_ax(%in_ax), %consts_slab <{
    num_q = 9223372036854775807 : i64,
    num_aux = 1 : i64,
    prime_offset = 0 : i64,
    pt_limbs = 4 : i64,
    pt_word_offsets = array<i64: 0>
  }> : (memref<4x16xi32>, memref<4x16xi32>, memref<4x16xi32>, memref<4x16xi32>, memref<1024xi8>) -> ()
  return
}
