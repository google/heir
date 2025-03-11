// RUN: heir-opt %s --convert-to-ciphertext-semantics=ciphertext-size=16 | FileCheck %s

#vec_layout = affine_map<(d0) -> (d0)>
#diagonal = affine_map<(d0, d1) -> (d1 mod 16, (d0 + d1) mod 16)>

// CHECK: [[row_major_indexing_map:#[^ ]*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: [[diagonal_layout:#[^ ]*]] = affine_map<(d0, d1) -> (d1 mod 16, (d0 + d1) mod 16)>

// CHECK-LABEL: @matvec_constant_matrix
// CHECK-SAME: [[arg0:%[^:]*]]: [[materialized_ty:!secret.secret<tensor<16xi16>>]]
// CHECK-SAME: tensor_ext.original_type = #tensor_ext.original_type<originalType = !secret.secret<tensor<16xi16>>, layout = (d0) -> (d0)>}
// CHECK-SAME: -> ([[materialized_ty]] {tensor_ext.original_type = #tensor_ext.original_type<originalType = !secret.secret<tensor<16xi16>>, layout = (d0) -> (d0)>})
func.func @matvec_constant_matrix(
    %arg0: !secret.secret<tensor<16xi16>> {tensor_ext.layout = #vec_layout}) ->
       (!secret.secret<tensor<16xi16>> {tensor_ext.layout = affine_map<(d0) -> (d0)>}) {
  // CHECK: [[cst:%[^ ]+]] = arith.constant dense<1> : tensor<16x16xi16>
  %cst = arith.constant dense<1> : tensor<16x16xi16>
  // CHECK: [[loop_init:%[^ ]+]] = arith.constant dense<0> : tensor<16xi16>
  %out = arith.constant dense<0> : tensor<16xi16>

  // CHECK: [[output:%[^ ]+]] = secret.generic
  // CHECK-SAME: ins([[arg0]]
  %0 = secret.generic ins(%arg0 : !secret.secret<tensor<16xi16>>)
                      attrs = {
                        __argattrs = [{tensor_ext.layout = #vec_layout}],
                        __resattrs = [{tensor_ext.layout = #vec_layout}]
                      } {
  // CHECK: ^body([[clear_arg0:%[^ ]+]]: tensor<16xi16>):
  ^body(%input0: tensor<16xi16>):
    // Nb. this op is just replaced with its input since the input happens to
    // already be materialized properly.
    %enc_out = tensor_ext.assign_layout %out {layout = #vec_layout, tensor_ext.layout = #vec_layout} : tensor<16xi16>

    // Apply the diagonal encoding
    // CHECK: [[enc_matrix_out:%[^ ]+]] = arith.constant dense<0> : tensor<16x16xi16>
    // CHECK: [[enc_matrix:%[^ ]+]] = linalg.generic
    // CHECK-SAME: indexing_maps = [
    // CHECK-SAME: [[row_major_indexing_map]],
    // CHECK-SAME: [[diagonal_layout]]]
    // CHECK-SAME: iterator_types = [
    // CHECK-SAME: "parallel", "parallel"]}
    // CHECK-NEXT: ^bb0([[in:%[^ ]+]]: i16, [[out:%[^ ]+]]: i16):
    // CHECK-NEXT: linalg.yield [[in]]
    %enc_matrix = tensor_ext.assign_layout %cst {layout = #diagonal, tensor_ext.layout = #diagonal} : tensor<16x16xi16>

    // Now the Halevi-Shoup kernel
    // CHECK: [[loop_output:%[^:]+]]:2 = affine.for [[induction:%[^ ]+]] = 0 to 16
    // CHECK-SAME: iter_args([[accum:%[^ ]+]] = [[loop_init]], [[inc_rotated:%[^ ]+]] = [[clear_arg0]])
    // CHECK-NEXT: [[c1:[^ ]*]] = arith.constant 1
    // CHECK-NEXT: [[rotated:[^ ]*]] = tensor_ext.rotate [[inc_rotated]], [[c1]]
    // CHECK-NEXT: tensor.extract_slice
    // CHECK-NEXT: [[collapsed:[^ ]*]] = tensor.collapse_shape
    // CHECK-NEXT: [[muled:[^ ]*]] = arith.muli [[rotated]], [[collapsed]]
    // CHECK-NEXT: [[added:[^ ]*]] = arith.addi [[accum]], [[muled]]
    // CHECK-NEXT: affine.yield [[added]], [[rotated]]
    %3 = linalg.matvec {tensor_ext.layout = #vec_layout}
          ins(%enc_matrix, %input0 : tensor<16x16xi16>, tensor<16xi16>)
          outs(%enc_out : tensor<16xi16>) -> tensor<16xi16>

    // CHECK: secret.yield [[loop_output]]#0
    secret.yield %3 : tensor<16xi16>
  } -> !secret.secret<tensor<16xi16>>
  // CHECK: return [[output]]
  return %0 : !secret.secret<tensor<16xi16>>
}
