// RUN: heir-opt --verify-diagnostics --split-input-file %s

#eval = #cheddar_gpu.form<eval>

// num_aux out of bounds for unary op
func.func @test_neg_num_aux_oob(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @below {{num_aux must be in [0, 4], but got 5}}
  cheddar_gpu.neg %out, %in <{
    in_form = #eval,
    out_form = #eval,
    num_aux = 5 : i64
  }> : memref<4x65536xi32>
  return
}
