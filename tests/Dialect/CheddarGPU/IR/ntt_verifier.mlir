// RUN: heir-opt --verify-diagnostics --split-input-file %s

// INTT epilogue without const buffer

func.func @test_intt_epilogue_without_const(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @below {{failed to verify that 'cst' is present iff epilogue is not 'none'}}
  cheddar_gpu.intt %out, %in <{
    montgomery_conversion = true,
    epilogue = #cheddar_gpu.intt_epilogue<mult_const>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  return
}

// -----
// INTT const buffer shape mismatch (5 != 4)

func.func @test_intt_const_shape_mismatch(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<5xi32>) {
  // expected-error @below {{constant buffer size (5) must match limb count (4)}}
  cheddar_gpu.intt %out, %in, cst(%cst) <{
    montgomery_conversion = true,
    epilogue = #cheddar_gpu.intt_epilogue<mult_const>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<5xi32>) -> ()
  return
}

// -----
// INTT none epilogue with unexpected const buffer

func.func @test_intt_none_epilogue_with_const(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<4xi32>) {
  // expected-error @below {{failed to verify that 'cst' is present iff epilogue is not 'none'}}
  cheddar_gpu.intt %out, %in, cst(%cst) <{
    montgomery_conversion = true,
    epilogue = #cheddar_gpu.intt_epilogue<none>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4xi32>) -> ()
  return
}

// -----
// INTT const buffer element type mismatch (CheddarGPU_OptionalElemTypeMatches)

func.func @test_intt_const_elem_type_mismatch(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<4xi64>) {
  // expected-error @below {{failed to verify that 'cst' (if present) has same element type as 'out'}}
  cheddar_gpu.intt %out, %in, cst(%cst) <{
    montgomery_conversion = true,
    epilogue = #cheddar_gpu.intt_epilogue<mult_const>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4xi64>) -> ()
  return
}

// -----
// INTT in_split split_count exceeds limb_count

func.func @test_intt_in_split_count_too_large(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @below {{in_split split_count must be in [0, limb_count]}}
  cheddar_gpu.intt %out, %in <{
    montgomery_conversion = true,
    epilogue = #cheddar_gpu.intt_epilogue<none>,
    in_split = array<i64: 5, 0>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  return
}

// -----
// INTT in_split input buffer too small for extent

func.func @test_intt_in_split_buffer_too_small(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @below {{input buffer too small for in_split extent}}
  cheddar_gpu.intt %out, %in <{
    montgomery_conversion = true,
    epilogue = #cheddar_gpu.intt_epilogue<none>,
    in_split = array<i64: 2, 3>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  return
}

// -----
// NTT in_limb_range and out_limb_range count mismatch

func.func @test_ntt_in_out_limb_range_mismatch(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @below {{limb count mismatch between in_limb_range and out_limb_range}}
  cheddar_gpu.ntt %out, %in <{
    montgomery_conversion = false,
    in_limb_range = array<i64: 0, 2>,
    out_limb_range = array<i64: 0, 3>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  return
}

// -----
// NTT prime_range count mismatch

func.func @test_ntt_prime_range_mismatch(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @below {{prime_range count mismatch with limb range count}}
  cheddar_gpu.ntt %out, %in <{
    montgomery_conversion = false,
    prime_range = array<i64: 0, 3>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  return
}

// -----
// NTT prime_range count compares against in_limb_range if set, else in's dim 0 (F8)
// Here only out_limb_range is set (count = 2), but in has 4 limbs, so prime_range needs 4 limbs.

func.func @test_ntt_prime_range_compares_against_in_limbs(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @below {{prime_range count mismatch with limb range count}}
  cheddar_gpu.ntt %out, %in <{
    montgomery_conversion = false,
    out_limb_range = array<i64: 0, 2>,
    prime_range = array<i64: 0, 2>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  return
}

// -----
// NTT skip_range start > end (CheddarGPU_SpanAttr)

func.func @test_ntt_skip_range_start_greater_than_end(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @below {{attribute 'skip_range' failed to satisfy constraint: i64 dense array attribute with exactly 2 elements of the form [start >= 0, end >= start]}}
  cheddar_gpu.ntt %out, %in <{
    montgomery_conversion = false,
    skip_range = array<i64: 3, 1>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  return
}

// -----
// NTT skip_range out of bounds (CheddarGPU_SpanEndWithinDim0)

func.func @test_ntt_skip_range_oob(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @below {{failed to verify that end of 'skip_range' is within dim 0 of 'out'}}
  cheddar_gpu.ntt %out, %in <{
    montgomery_conversion = false,
    skip_range = array<i64: 1, 5>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  return
}

// -----
// NTT skip_copy_src without skip_range (CheddarGPU_Requires)

func.func @test_ntt_skip_copy_src_needs_skip_range(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %s: memref<4x65536xi32>) {
  // expected-error @below {{failed to verify that 'skip_copy_src' requires 'skip_range' to be set}}
  cheddar_gpu.ntt %out, %in, skip_copy_src(%s) <{
    montgomery_conversion = true
  }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  return
}

// -----
// NTT skip_copy_src degree mismatch (CheddarGPU_OptionalSameRingAs)

func.func @test_ntt_skip_copy_src_degree_mismatch(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %s: memref<4x32768xi32>) {
  // expected-error @+1 {{failed to verify that 'skip_copy_src' (if present) has the degree and word type of 'out'}}
  cheddar_gpu.ntt %out, %in, skip_copy_src(%s) <{
    montgomery_conversion = true,
    skip_range = array<i64: 0, 2>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4x32768xi32>) -> ()
  return
}

// -----
// NTT skip_copy_src buffer too small

func.func @test_ntt_skip_copy_src_too_small(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %skip: memref<1x65536xi32>) {
  // expected-error @+1 {{skip_copy_src buffer too small for skipped range}}
  cheddar_gpu.ntt %out, %in, skip_copy_src(%skip) <{
    montgomery_conversion = false,
    skip_range = array<i64: 1, 3>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<1x65536xi32>) -> ()
  return
}
