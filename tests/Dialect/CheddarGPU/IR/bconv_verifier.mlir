// RUN: heir-opt --verify-diagnostics --split-input-file %s

// -----
// Wrong input form (coeff instead of coeff_centered)

func.func @test_wrong_in_form(%out: memref<3x65536xi32>, %in: memref<4x65536xi32>, %table: memref<3x4xi32>) {
  // expected-error @+1 {{bconv requires coeff_centered input form}}
  cheddar_gpu.bconv %out, %in, %table <{
    in_form = #cheddar_gpu.form<coeff>,
    out_form = #cheddar_gpu.form<coeff>
  }> : memref<3x65536xi32>, memref<4x65536xi32>, memref<3x4xi32>
  return
}

// -----
// Wrong output form (eval instead of coeff)

func.func @test_wrong_out_form(%out: memref<3x65536xi32>, %in: memref<4x65536xi32>, %table: memref<3x4xi32>) {
  // expected-error @+1 {{bconv requires coeff output form}}
  cheddar_gpu.bconv %out, %in, %table <{
    in_form = #cheddar_gpu.form<coeff_centered>,
    out_form = #cheddar_gpu.form<eval>
  }> : memref<3x65536xi32>, memref<4x65536xi32>, memref<3x4xi32>
  return
}

// -----
// Element type mismatch between out and table

func.func @test_elem_type_mismatch(%out: memref<3x65536xi32>, %in: memref<4x65536xi32>, %table: memref<3x4xi64>) {
  // expected-error @+1 {{element type mismatch between out, in, and table}}
  cheddar_gpu.bconv %out, %in, %table <{
    in_form = #cheddar_gpu.form<coeff_centered>,
    out_form = #cheddar_gpu.form<coeff>
  }> : memref<3x65536xi32>, memref<4x65536xi32>, memref<3x4xi64>
  return
}

// -----
// Rank mismatch (table is rank 1 instead of 2)

func.func @test_table_rank(%out: memref<3x65536xi32>, %in: memref<4x65536xi32>, %table: memref<12xi32>) {
  // expected-error @+1 {{table must have rank 2}}
  cheddar_gpu.bconv %out, %in, %table <{
    in_form = #cheddar_gpu.form<coeff_centered>,
    out_form = #cheddar_gpu.form<coeff>
  }> : memref<3x65536xi32>, memref<4x65536xi32>, memref<12xi32>
  return
}

// -----
// Degree mismatch between out and in

func.func @test_degree_mismatch(%out: memref<3x32768xi32>, %in: memref<4x65536xi32>, %table: memref<3x4xi32>) {
  // expected-error @+1 {{polynomial degree mismatch between out (32768) and in (65536)}}
  cheddar_gpu.bconv %out, %in, %table <{
    in_form = #cheddar_gpu.form<coeff_centered>,
    out_form = #cheddar_gpu.form<coeff>
  }> : memref<3x32768xi32>, memref<4x65536xi32>, memref<3x4xi32>
  return
}

// -----
// Table source length mismatch with active input limbs

func.func @test_table_src_len_mismatch(%out: memref<3x65536xi32>, %in: memref<4x65536xi32>, %table: memref<3x5xi32>) {
  // expected-error @+1 {{table source length (5) must match input limb count (4)}}
  cheddar_gpu.bconv %out, %in, %table <{
    in_form = #cheddar_gpu.form<coeff_centered>,
    out_form = #cheddar_gpu.form<coeff>
  }> : memref<3x65536xi32>, memref<4x65536xi32>, memref<3x5xi32>
  return
}

// -----
// Table destination length mismatch with active output limbs

func.func @test_table_dst_len_mismatch(%out: memref<3x65536xi32>, %in: memref<4x65536xi32>, %table: memref<4x4xi32>) {
  // expected-error @+1 {{table destination length (4) must match active destination limbs (3)}}
  cheddar_gpu.bconv %out, %in, %table <{
    in_form = #cheddar_gpu.form<coeff_centered>,
    out_form = #cheddar_gpu.form<coeff>
  }> : memref<3x65536xi32>, memref<4x65536xi32>, memref<4x4xi32>
  return
}

// -----
// Skip range out of bounds

func.func @test_skip_range_oob(%out: memref<6x65536xi32>, %in: memref<4x65536xi32>, %table: memref<4x4xi32>) {
  // expected-error @+1 {{skip_range [2, 7] out of bounds for active destination with 6 limbs}}
  cheddar_gpu.bconv %out, %in, %table <{
    in_form = #cheddar_gpu.form<coeff_centered>,
    out_form = #cheddar_gpu.form<coeff>,
    skip_range = array<i64: 2, 7>
  }> : memref<6x65536xi32>, memref<4x65536xi32>, memref<4x4xi32>
  return
}

// -----
// Table destination length mismatch with skipped limbs

func.func @test_table_dst_with_skip_mismatch(%out: memref<6x65536xi32>, %in: memref<4x65536xi32>, %table: memref<5x4xi32>) {
  // expected-error @+1 {{table destination length (5) must match active destination limbs minus skipped limbs (4)}}
  cheddar_gpu.bconv %out, %in, %table <{
    in_form = #cheddar_gpu.form<coeff_centered>,
    out_form = #cheddar_gpu.form<coeff>,
    skip_range = array<i64: 2, 4>
  }> : memref<6x65536xi32>, memref<4x65536xi32>, memref<5x4xi32>
  return
}
