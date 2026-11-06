// RUN: heir-opt --verify-diagnostics --split-input-file %s

// in_limb_range out of bounds for `in` (CheddarGPU_InOutLimbRangesWithin)

func.func @bconv_in_range_oob(%out: memref<3x65536xi32>, %in: memref<4x65536xi32>, %table: memref<3x4xi32>) {
  // expected-error @below {{limb range [3, 2] out of bounds for memref with 4 limbs}}
  cheddar_gpu.bconv %out, %in, %table <{
    in_limb_range = array<i64: 3, 2>
  }> : (memref<3x65536xi32>, memref<4x65536xi32>, memref<3x4xi32>) -> ()
  return
}

// -----
// Table source length mismatch with active input limbs

func.func @test_table_src_len_mismatch(%out: memref<3x65536xi32>, %in: memref<4x65536xi32>, %table: memref<3x5xi32>) {
  // expected-error @below {{table source length (5) must match input limb count (4)}}
  cheddar_gpu.bconv %out, %in, %table : (memref<3x65536xi32>, memref<4x65536xi32>, memref<3x5xi32>) -> ()
  return
}

// -----
// Table destination length mismatch with active output limbs

func.func @test_table_dst_len_mismatch(%out: memref<3x65536xi32>, %in: memref<4x65536xi32>, %table: memref<4x4xi32>) {
  // expected-error @below {{table destination length (4) must match active destination limbs (3)}}
  cheddar_gpu.bconv %out, %in, %table : (memref<3x65536xi32>, memref<4x65536xi32>, memref<4x4xi32>) -> ()
  return
}

// -----
// Skip range out of bounds

func.func @test_skip_range_oob(%out: memref<6x65536xi32>, %in: memref<4x65536xi32>, %table: memref<4x4xi32>) {
  // expected-error @below {{skip_range [2, 7] out of bounds for active destination with 6 limbs}}
  cheddar_gpu.bconv %out, %in, %table <{
    skip_range = array<i64: 2, 7>
  }> : (memref<6x65536xi32>, memref<4x65536xi32>, memref<4x4xi32>) -> ()
  return
}

// -----
// Table destination length mismatch with skipped limbs

func.func @test_table_dst_with_skip_mismatch(%out: memref<6x65536xi32>, %in: memref<4x65536xi32>, %table: memref<5x4xi32>) {
  // expected-error @below {{table destination length (5) must match active destination limbs minus skipped limbs (4)}}
  cheddar_gpu.bconv %out, %in, %table <{
    skip_range = array<i64: 2, 4>
  }> : (memref<6x65536xi32>, memref<4x65536xi32>, memref<5x4xi32>) -> ()
  return
}

// -----
// BConv dst_primes length mismatch with table destination length

func.func @test_bconv_dst_primes_mismatch(%out: memref<3x65536xi32>, %in: memref<4x65536xi32>, %table: memref<3x4xi32>) {
  // expected-error @below {{dst_primes length mismatch with table destination length}}
  cheddar_gpu.bconv %out, %in, %table <{
    dst_primes = array<i64: 1, 2>
  }> : (memref<3x65536xi32>, memref<4x65536xi32>, memref<3x4xi32>) -> ()
  return
}
