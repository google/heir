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
// num_aux out of bounds on tensor operands

func.func @test_num_aux_oob_tensor(%out: tensor<4x65536xi32>, %lhs: tensor<4x65536xi32>, %rhs: tensor<4x65536xi32>) -> tensor<4x65536xi32> {
  // expected-error @below {{num_aux must be in [0, 4], but got 5}}
  %res = cheddar_gpu.add %out, %lhs, %rhs <{
    lhs_form = #cheddar_gpu.form<eval>,
    rhs_form = #cheddar_gpu.form<eval>,
    out_form = #cheddar_gpu.form<eval>,
    num_aux = 5 : i64
  }> : tensor<4x65536xi32> -> tensor<4x65536xi32>
  return %res : tensor<4x65536xi32>
}

// -----
// num_aux equal to the limb count is accepted on tensor operands

func.func @test_num_aux_at_bound_tensor(%out: tensor<4x65536xi32>, %lhs: tensor<4x65536xi32>, %rhs: tensor<4x65536xi32>) -> tensor<4x65536xi32> {
  %res = cheddar_gpu.add %out, %lhs, %rhs <{
    lhs_form = #cheddar_gpu.form<eval>,
    rhs_form = #cheddar_gpu.form<eval>,
    out_form = #cheddar_gpu.form<eval>,
    num_aux = 4 : i64
  }> : tensor<4x65536xi32> -> tensor<4x65536xi32>
  return %res : tensor<4x65536xi32>
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
// -----
// Tensor-mode add with no result type

#eval = #cheddar_gpu.form<eval>

func.func @test_tensor_no_result(%out: tensor<4x65536xi32>, %lhs: tensor<4x65536xi32>, %rhs: tensor<4x65536xi32>) {
  // expected-error @below {{expected the number of tensor results (0) to be equal to the number of output tensors (1)}}
  cheddar_gpu.add %out, %lhs, %rhs <{
    lhs_form = #eval,
    rhs_form = #eval,
    out_form = #eval
  }> : tensor<4x65536xi32>
  return
}

// -----
// Mixed tensor and memref operands

#eval = #cheddar_gpu.form<eval>

// expected-note @below {{prior use here}}
func.func @test_mixed_operands(%out: memref<4x65536xi32>, %lhs: tensor<4x65536xi32>, %rhs: memref<4x65536xi32>) {
  // expected-error @below {{use of value '%lhs' expects different type than prior uses: 'memref<4x65536xi32>' vs 'tensor<4x65536xi32>'}}
  cheddar_gpu.add %out, %lhs, %rhs <{
    lhs_form = #eval,
    rhs_form = #eval,
    out_form = #eval
  }> : memref<4x65536xi32>
  return
}

// -----
// Mixed tensor and memref operands (generic syntax)

#eval = #cheddar_gpu.form<eval>

func.func @test_mixed_operands_generic(%out: memref<4x65536xi32>, %lhs: tensor<4x65536xi32>, %rhs: memref<4x65536xi32>) {
  // expected-error @below {{failed to verify that all of {out, lhs, rhs} have same type}}
  "cheddar_gpu.add"(%out, %lhs, %rhs) <{
    lhs_form = #eval,
    rhs_form = #eval,
    out_form = #eval
  }> : (memref<4x65536xi32>, tensor<4x65536xi32>, memref<4x65536xi32>) -> ()
  return
}
