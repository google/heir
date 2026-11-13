// RUN: heir-opt --verify-diagnostics --split-input-file %s

// Form mismatch on add (lhs coeff, rhs eval)

func.func @test_form_mismatch_add(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
  // expected-error @below {{failed to verify that all of {lhs_form, rhs_form, out_form} have same form}}
  cheddar_gpu.add %out, %lhs, %rhs <{
    lhs_form = #cheddar_gpu.form<coeff>,
    rhs_form = #cheddar_gpu.form<eval>,
    out_form = #cheddar_gpu.form<coeff>
  }> : memref<4x65536xi32>
  return
}

// -----
// num_aux out of bounds

func.func @test_num_aux_oob(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
  // expected-error @below {{num_aux must be in [0, 4], but got 5}}
  cheddar_gpu.add %out, %lhs, %rhs <{
    lhs_form = #cheddar_gpu.form<eval>,
    rhs_form = #cheddar_gpu.form<eval>,
    out_form = #cheddar_gpu.form<eval>,
    num_aux = 5 : i64
  }> : memref<4x65536xi32>
  return
}

// -----
// Pair op with out_limb_range out of bounds

func.func @test_pair_add_out_range_oob(
    %out_bx: memref<4x65536xi32>, %out_ax: memref<4x65536xi32>,
    %lhs_bx: memref<4x65536xi32>, %lhs_ax: memref<4x65536xi32>,
    %rhs_bx: memref<4x65536xi32>, %rhs_ax: memref<4x65536xi32>) {
  // expected-error @below {{limb range [3, 2] out of bounds for memref with 4 limbs}}
  cheddar_gpu.add_2 ins(%lhs_bx, %lhs_ax, %rhs_bx, %rhs_ax)
                    outs(%out_bx, %out_ax) <{
    lhs_form = #cheddar_gpu.form<eval>,
    rhs_form = #cheddar_gpu.form<eval>,
    out_form = #cheddar_gpu.form<eval>,
    out_limb_range = array<i64: 3, 2>
  }> : memref<4x65536xi32>
  return
}
