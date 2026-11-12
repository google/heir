// RUN: heir-opt --verify-diagnostics --split-input-file %s

// Degree mismatch on copy

func.func @test_copy_degree_mismatch(%out: memref<4x32768xi32>, %in: memref<4x65536xi32>) {
  // expected-error @below {{failed to verify that all of {out, in} have same polynomial degree (dim 1)}}
  cheddar_gpu.copy %out, %in : (memref<4x32768xi32>, memref<4x65536xi32>) -> ()
  return
}

// -----
// Copy in_limb_range out of bounds

func.func @test_copy_in_limb_range_oob(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @below {{limb range [2, 3] out of bounds for memref with 4 limbs}}
  cheddar_gpu.copy %out, %in <{
    in_limb_range = array<i64: 2, 3>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  return
}

// -----
// Copy out_limb_range out of bounds

func.func @test_copy_out_limb_range_oob(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @below {{limb range [3, 2] out of bounds for memref with 4 limbs}}
  cheddar_gpu.copy %out, %in <{
    out_limb_range = array<i64: 3, 2>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  return
}

// -----
// Copy limb count mismatch between in and out

func.func @test_copy_limb_count_mismatch(%out: memref<4x65536xi32>, %in: memref<2x65536xi32>) {
  // expected-error @below {{limb count mismatch between in and out}}
  cheddar_gpu.copy %out, %in : (memref<4x65536xi32>, memref<2x65536xi32>) -> ()
  return
}

// -----
// Copy explicit limb range count mismatch between in and out

func.func @test_copy_explicit_limb_range_count_mismatch(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @below {{limb count mismatch between in and out (2 vs 3)}}
  cheddar_gpu.copy %out, %in <{
    in_limb_range = array<i64: 0, 2>,
    out_limb_range = array<i64: 0, 3>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>) -> ()
  return
}
