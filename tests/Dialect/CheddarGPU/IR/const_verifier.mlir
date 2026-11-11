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
