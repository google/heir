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
// Limb range invalid offset/count (offset < 0)

func.func @test_limb_range_negative_offset(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
  // expected-error @below {{failed to satisfy constraint: i64 dense array attribute with exactly 2 elements of the form [offset >= 0, count > 0]}}
  cheddar_gpu.add %out, %lhs, %rhs <{
    lhs_form = #cheddar_gpu.form<eval>,
    rhs_form = #cheddar_gpu.form<eval>,
    out_form = #cheddar_gpu.form<eval>,
    lhs_limb_range = array<i64: -1, 2>
  }> : memref<4x65536xi32>
  return
}

// -----
// Limb range out of bounds (offset 2 + count 3 = 5 > 4)

func.func @test_limb_range_oob(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
  // expected-error @below {{limb range [2, 3] out of bounds for memref with 4 limbs}}
  cheddar_gpu.add %out, %lhs, %rhs <{
    lhs_form = #cheddar_gpu.form<eval>,
    rhs_form = #cheddar_gpu.form<eval>,
    out_form = #cheddar_gpu.form<eval>,
    lhs_limb_range = array<i64: 2, 3>
  }> : memref<4x65536xi32>
  return
}

// -----
// Limb range exceeds module params limb count

module attributes {
  cheddar_gpu.params = #cheddar_gpu.params<word = 32, log_n = 16, ter = [1000], main = [2000, 3000], aux = [4000], level_config = [[[2 : i32, 1 : i32]]], alpha = 1, dnum = 0>
} {
  func.func @test_limb_range_exceeds_module_params(%out: memref<6x65536xi32>, %lhs: memref<6x65536xi32>, %rhs: memref<6x65536xi32>) {
    // expected-error @below {{limb range [0, 5] exceeds module params limb count (4)}}
    cheddar_gpu.add %out, %lhs, %rhs <{
      lhs_form = #cheddar_gpu.form<eval>,
      rhs_form = #cheddar_gpu.form<eval>,
      out_form = #cheddar_gpu.form<eval>,
      lhs_limb_range = array<i64: 0, 5>
    }> : memref<6x65536xi32>
    return
  }
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
// Mul with coeff operand (requires eval)

func.func @test_mul_with_coeff(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
  // expected-error @below {{failed to verify that {lhs_form, rhs_form, out_form} are 'eval'}}
  cheddar_gpu.mul %out, %lhs, %rhs <{
    lhs_form = #cheddar_gpu.form<coeff>,
    rhs_form = #cheddar_gpu.form<eval>,
    out_form = #cheddar_gpu.form<eval>
  }> : memref<4x65536xi32>
  return
}
