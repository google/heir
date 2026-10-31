// RUN: heir-opt --verify-diagnostics --split-input-file %s

// -----
// Form mismatch on add (lhs coeff, rhs eval)

func.func @test_form_mismatch_add(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
  // expected-error @+1 {{form mismatch on add: lhs form (coeff) and rhs form (eval) must match}}
  cheddar_gpu.add %out, %lhs, %rhs <{
    lhs_form = #cheddar_gpu.form<coeff>,
    rhs_form = #cheddar_gpu.form<eval>,
    out_form = #cheddar_gpu.form<coeff>
  }> : memref<4x65536xi32>
  return
}

// -----
// Mul with coeff operand (requires eval)

func.func @test_mul_with_coeff(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
  // expected-error @+1 {{mul requires eval form for all operands}}
  cheddar_gpu.mul %out, %lhs, %rhs <{
    lhs_form = #cheddar_gpu.form<coeff>,
    rhs_form = #cheddar_gpu.form<eval>,
    out_form = #cheddar_gpu.form<eval>
  }> : memref<4x65536xi32>
  return
}

// -----
// Shape mismatch between operands

func.func @test_shape_mismatch(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<5x65536xi32>) {
  // expected-error @+1 {{failed to verify that all of {out, lhs, rhs} have same type}}
  "cheddar_gpu.add"(%out, %lhs, %rhs) <{
    lhs_form = #cheddar_gpu.form<eval>,
    rhs_form = #cheddar_gpu.form<eval>,
    out_form = #cheddar_gpu.form<eval>
  }> : (memref<4x65536xi32>, memref<4x65536xi32>, memref<5x65536xi32>) -> ()
  return
}

// -----
// Bad word width (i16 instead of i32/i64)

func.func @test_bad_word_width(%out: memref<4x65536xi16>, %lhs: memref<4x65536xi16>, %rhs: memref<4x65536xi16>) {
  // expected-error @+1 {{element type must be i32 or i64, but got}}
  cheddar_gpu.add %out, %lhs, %rhs <{
    lhs_form = #cheddar_gpu.form<eval>,
    rhs_form = #cheddar_gpu.form<eval>,
    out_form = #cheddar_gpu.form<eval>
  }> : memref<4x65536xi16>
  return
}

// -----
// Limb range out of bounds (offset 2 + count 3 = 5 > 4)

func.func @test_limb_range_oob(%out: memref<4x65536xi32>, %lhs: memref<4x65536xi32>, %rhs: memref<4x65536xi32>) {
  // expected-error @+1 {{limb range [2, 3] out of bounds for memref with 4 limbs}}
  cheddar_gpu.add %out, %lhs, %rhs <{
    lhs_form = #cheddar_gpu.form<eval>,
    rhs_form = #cheddar_gpu.form<eval>,
    out_form = #cheddar_gpu.form<eval>,
    lhs_limb_range = array<i64: 2, 3>
  }> : memref<4x65536xi32>
  return
}

// -----
// Permute on coeff form (requires eval)

func.func @test_permute_coeff(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @+1 {{permute requires eval form for all operands}}
  cheddar_gpu.permute %out, %in <{
    galois = 5 : i64,
    in_form = #cheddar_gpu.form<coeff>,
    out_form = #cheddar_gpu.form<eval>
  }> : memref<4x65536xi32>
  return
}

// -----
// Permute missing both galois and conj

func.func @test_permute_missing_galois(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @+1 {{permute requires either galois or conj attribute}}
  cheddar_gpu.permute %out, %in <{
    in_form = #cheddar_gpu.form<eval>,
    out_form = #cheddar_gpu.form<eval>
  }> : memref<4x65536xi32>
  return
}

// -----
// Permute with both galois and conj

func.func @test_permute_both_galois_conj(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @+1 {{cannot specify both galois and conj attributes}}
  cheddar_gpu.permute %out, %in <{
    galois = 5 : i64,
    conj,
    in_form = #cheddar_gpu.form<eval>,
    out_form = #cheddar_gpu.form<eval>
  }> : memref<4x65536xi32>
  return
}

// -----
// NTT with eval input (requires coeff)

func.func @test_ntt_with_eval_input(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @+1 {{ntt requires coeff form for input}}
  cheddar_gpu.ntt %out, %in <{
    in_form = #cheddar_gpu.form<eval>,
    out_form = #cheddar_gpu.form<eval>,
    montgomery_conversion = false
  }> : memref<4x65536xi32>
  return
}

// -----
// INTT with coeff input (requires eval)

func.func @test_intt_with_coeff_input(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @+1 {{intt requires eval form for input}}
  cheddar_gpu.intt %out, %in <{
    in_form = #cheddar_gpu.form<coeff>,
    out_form = #cheddar_gpu.form<coeff>,
    montgomery_conversion = true,
    epilogue = #cheddar_gpu.intt_epilogue<none>
  }> : memref<4x65536xi32>
  return
}

// -----
// INTT epilogue without const buffer

func.func @test_intt_epilogue_without_const(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>) {
  // expected-error @+1 {{intt with mult_const epilogue requires constant buffer}}
  cheddar_gpu.intt %out, %in <{
    in_form = #cheddar_gpu.form<eval>,
    out_form = #cheddar_gpu.form<coeff>,
    montgomery_conversion = true,
    epilogue = #cheddar_gpu.intt_epilogue<mult_const>
  }> : memref<4x65536xi32>
  return
}

// -----
// INTT const buffer shape mismatch (5 != 4)

func.func @test_intt_const_shape_mismatch(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<5xi32>) {
  // expected-error @+1 {{constant buffer size (5) must match limb count (4)}}
  cheddar_gpu.intt %out, %in, %cst <{
    in_form = #cheddar_gpu.form<eval>,
    out_form = #cheddar_gpu.form<coeff>,
    montgomery_conversion = true,
    epilogue = #cheddar_gpu.intt_epilogue<mult_const>
  }> : memref<4x65536xi32>, memref<5xi32>
  return
}

// -----
// INTT mult_const_normalize with coeff out_form (requires coeff_centered)

func.func @test_intt_mult_const_normalize_with_coeff(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<4xi32>) {
  // expected-error @+1 {{intt with mult_const_normalize epilogue requires coeff_centered output form}}
  cheddar_gpu.intt %out, %in, %cst <{
    in_form = #cheddar_gpu.form<eval>,
    out_form = #cheddar_gpu.form<coeff>,
    montgomery_conversion = true,
    epilogue = #cheddar_gpu.intt_epilogue<mult_const_normalize>
  }> : memref<4x65536xi32>, memref<4xi32>
  return
}

// -----
// INTT none epilogue with unexpected const buffer

func.func @test_intt_none_epilogue_with_const(%out: memref<4x65536xi32>, %in: memref<4x65536xi32>, %cst: memref<4xi32>) {
  // expected-error @+1 {{intt with none epilogue does not accept constant buffer}}
  cheddar_gpu.intt %out, %in, %cst <{
    in_form = #cheddar_gpu.form<eval>,
    out_form = #cheddar_gpu.form<coeff>,
    montgomery_conversion = true,
    epilogue = #cheddar_gpu.intt_epilogue<none>
  }> : memref<4x65536xi32>, memref<4xi32>
  return
}
