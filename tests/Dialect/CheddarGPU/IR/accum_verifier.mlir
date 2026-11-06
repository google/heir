// RUN: heir-opt --split-input-file --verify-diagnostics %s

// Permute missing both galois and conj (CheddarGPU_ExactlyOneOf)

func.func @test_permute_missing_galois(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @below {{failed to verify that exactly one of 'galois' and 'conj' is set}}
  cheddar_gpu.permute %out, %in : memref<4x65536xi32>
  return
}

// -----
// Permute with both galois and conj (CheddarGPU_ExactlyOneOf)

func.func @test_permute_both_galois_conj(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @below {{failed to verify that exactly one of 'galois' and 'conj' is set}}
  cheddar_gpu.permute %out, %in <{
    galois = 5 : i64,
    conj
  }> : memref<4x65536xi32>
  return
}

// -----
// PermuteAccum requires at least one source (CheddarGPU_NonEmpty)

func.func @test_permute_accum_empty_srcs(%out: memref<4x65536xi32>) {
  // expected-error @below {{failed to verify that 'srcs' is non-empty}}
  cheddar_gpu.permute_accum %out, srcs() <{
    galois_factors = array<i64>
  }> : (memref<4x65536xi32>) -> ()
  return
}

// -----
// PermuteAccum srcs type mismatch with out (CheddarGPU_EachTypeIs)

func.func @test_permute_accum_srcs_type_mismatch(%out: memref<4x65536xi32>, %s: memref<2x65536xi32>) {
  // expected-error @below {{failed to verify that every 'srcs' has the type of 'out'}}
  cheddar_gpu.permute_accum %out, srcs(%s) <{
    galois_factors = array<i64: 5>
  }> : (memref<4x65536xi32>, memref<2x65536xi32>) -> ()
  return
}

// -----
// CPAccum empty ct_srcs (CheddarGPU_NonEmpty)

func.func @test_cp_accum_empty_ct_srcs(%out: memref<4x65536xi32>) {
  // expected-error @below {{failed to verify that 'ct_srcs' is non-empty}}
  cheddar_gpu.cp_accum %out, ct_srcs(), common_srcs() : (memref<4x65536xi32>) -> ()
  return
}

// -----
// CPAccum count mismatch between ct_srcs and common_srcs (CheddarGPU_SameCount)

func.func @test_cp_accum_src_count_mismatch(%out: memref<4x65536xi32>, %c0: memref<4x65536xi32>) {
  // expected-error @below {{failed to verify that all of {ct_srcs, common_srcs} have same count}}
  cheddar_gpu.cp_accum %out, ct_srcs(%c0), common_srcs() : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  return
}

// -----
// CPAccum ct_srcs ring mismatch with out (CheddarGPU_EachSameRingAs)

func.func @test_cp_accum_ct_srcs_ring_mismatch(%out: memref<4x65536xi32>, %c: memref<4x32768xi32>, %p: memref<4x65536xi32>) {
  // expected-error @below {{failed to verify that every 'ct_srcs' has the degree and word type of 'out'}}
  cheddar_gpu.cp_accum %out, ct_srcs(%c), common_srcs(%p) : (memref<4x65536xi32>, memref<4x32768xi32>, memref<4x65536xi32>) -> ()
  return
}

// -----
// CPAccum 2D common_srcs type mismatch with out (CheddarGPU_Each2DTypeIs)

func.func @test_cp_accum_2d_common_type_mismatch(%out: memref<4x65536xi32>, %c: memref<4x65536xi32>, %p: memref<2x65536xi32>) {
  // expected-error @below {{failed to verify that every 2D 'common_srcs' has the type of 'out'}}
  cheddar_gpu.cp_accum %out, ct_srcs(%c), common_srcs(%p) : (memref<4x65536xi32>, memref<4x65536xi32>, memref<2x65536xi32>) -> ()
  return
}

// -----
// CPAccum ct_src limb count smaller than outCount (CheddarGPU_EachCoversOutCount)

func.func @test_cp_accum_ct_src_limb_count_too_small(%out: memref<4x65536xi32>, %c0: memref<2x65536xi32>, %p0: memref<4x65536xi32>) {
  // expected-error @below {{failed to verify that every 'ct_srcs' has at least as many limbs as the active range of 'out'}}
  cheddar_gpu.cp_accum %out, ct_srcs(%c0), common_srcs(%p0) : (memref<4x65536xi32>, memref<2x65536xi32>, memref<4x65536xi32>) -> ()
  return
}

// -----
// CPAccum 1D common_src size smaller than outCount (CheddarGPU_EachCoversOutCount)

func.func @test_cp_accum_1d_common_size_too_small(%out: memref<4x65536xi32>, %c: memref<4x65536xi32>, %v: memref<2xi32>) {
  // expected-error @below {{failed to verify that every 1D 'common_srcs' has at least as many limbs as the active range of 'out'}}
  cheddar_gpu.cp_accum %out, ct_srcs(%c), common_srcs(%v) : (memref<4x65536xi32>, memref<4x65536xi32>, memref<2xi32>) -> ()
  return
}

// -----
// out_limb_range is [offset, count]: [1, 3] needs ct_src with >= 3 limbs.
// The old range[1]-range[0] computation (= 2) would have accepted this.

func.func @test_cp_accum_limb_range_is_offset_count(%out: memref<4x65536xi32>, %c0: memref<2x65536xi32>, %p0: memref<4x65536xi32>) {
  // expected-error @below {{failed to verify that every 'ct_srcs' has at least as many limbs as the active range of 'out'}}
  cheddar_gpu.cp_accum %out, ct_srcs(%c0), common_srcs(%p0) <{
    out_limb_range = array<i64: 1, 3>
  }> : (memref<4x65536xi32>, memref<2x65536xi32>, memref<4x65536xi32>) -> ()
  return
}

// -----
// PermuteAccum2 requires both acc_bx and acc_ax or neither (CheddarGPU_BothOrNeither)

func.func @test_permute_accum_2_acc_both_or_neither(%out_b: memref<4x65536xi32>, %out_a: memref<4x65536xi32>,
                                                    %acc_b: memref<4x65536xi32>,
                                                    %sb0: memref<4x65536xi32>, %sa0: memref<4x65536xi32>) {
  // expected-error @below {{failed to verify that 'acc_bx' and 'acc_ax' are both present or both absent}}
  "cheddar_gpu.permute_accum_2"(%out_b, %out_a, %acc_b, %sb0, %sa0) <{
    galois_factors = array<i64: 5>,
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 1>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  return
}
