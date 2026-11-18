// RUN: heir-opt --verify-diagnostics --split-input-file %s

// Constant buffer size mismatch with limb count

func.func @test_add_const_cst_dim_mismatch(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<5xi32>) {
  // expected-error @below {{failed to verify that 'cst' has one entry per limb of 'out'}}
  cheddar_gpu.add_const %out, %in, %cst <{
    in_form = #cheddar_gpu.form<eval>,
    out_form = #cheddar_gpu.form<eval>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<5xi32>) -> ()
  return
}

// -----
// in_limb_range out of bounds for `in`

func.func @test_add_const_in_range_oob(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<4xi32>) {
  // expected-error @below {{limb range [3, 2] out of bounds for memref with 4 limbs}}
  cheddar_gpu.add_const %out, %in, %cst <{
    in_form = #cheddar_gpu.form<eval>,
    out_form = #cheddar_gpu.form<eval>,
    in_limb_range = array<i64: 3, 2>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<4xi32>) -> ()
  return
}
