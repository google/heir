// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics=ciphertext-size=16 | FileCheck %s

#vec_layout = #tensor_ext.layout<map = (d0) -> (d0 mod 16)>
#diagonal = #tensor_ext.layout<map = (d0, d1) -> ((d1 - d0) mod 16, (d1 - (d1 - d0) mod 16) mod 16)>

// CHECK-DAG: [[row_major_indexing_map:#[^ ]*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: [[diagonal_layout:#[^ ]*]] = affine_map<(d0, d1) -> ((d1 - d0) mod 16, (d1 - (d1 - d0) mod 16) mod 16)>
// CHECK-DAG: [[layout_rm_1d:#[^ ]*]] = #tensor_ext.layout<map = (d0) -> (d0 mod 16)>
// CHECK-DAG: [[orig_type:#[^ ]*]] = #tensor_ext.original_type<originalType = tensor<16xi16>, layout = [[layout_rm_1d]]>

// CHECK: @matvec_constant_matrix
// CHECK-SAME: [[arg0:%[^:]*]]: [[materialized_ty:!secret.secret<tensor<16xi16>>]]
// CHECK-SAME: tensor_ext.original_type = [[orig_type]]
// CHECK-SAME: -> ([[materialized_ty]] {tensor_ext.original_type = [[orig_type]]}
func.func @matvec_constant_matrix(
    %arg0: !secret.secret<tensor<16xi16>> {tensor_ext.layout = #vec_layout}) ->
       (!secret.secret<tensor<16xi16>> {tensor_ext.layout = #vec_layout}) {
  // CHECK: [[cst:%[^ ]+]] = arith.constant dense<1> : tensor<16x16xi16>
  %cst = arith.constant dense<1> : tensor<16x16xi16>
  // CHECK: [[loop_init:%[^ ]+]] = arith.constant dense<0> : tensor<16xi16>
  %out = arith.constant dense<0> : tensor<16xi16>

  // CHECK: [[output:%[^ ]+]] = secret.generic
  // CHECK-SAME: ([[arg0]]
  %0 = secret.generic(%arg0 : !secret.secret<tensor<16xi16>>)
                      attrs = {
                        __argattrs = [{tensor_ext.layout = #vec_layout}],
                        __resattrs = [{tensor_ext.layout = #vec_layout}]
                      } {
  // CHECK: ^body([[clear_arg0:%[^ ]+]]: tensor<16xi16>):
  ^body(%input0: tensor<16xi16>):
    // Apply the row-major encoding
    // CHECK: [[enc_vec_out:%[^ ]+]] = arith.constant dense<0> : tensor<16xi16>
    // CHECK: [[enc_vec:%[^ ]+]] = linalg.generic
    // CHECK-SAME: [[loop_init]]
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

    // 16 iterations, unrolled
    // CHECK: [[c0:[^ ]*]] = arith.constant 0
    // CHECK-NEXT: [[rotated:[^ ]*]] = tensor_ext.rotate [[clear_arg0]], [[c0]]
    // CHECK-NEXT: [[row:[^ ]*]] = tensor.extract_slice
    // CHECK-NEXT: [[muled:[^ ]*]] = arith.muli [[rotated]], [[row]]
    // CHECK-NEXT: [[added:[^ ]*]] = arith.addi [[enc_vec]], [[muled]]

    // CHECK: [[c1:[^ ]*]] = arith.constant 1
    // CHECK-NEXT: tensor_ext.rotate [[clear_arg0]], [[c1]]
    // CHECK-NEXT: tensor.extract_slice
    // CHECK-NEXT: arith.muli
    // CHECK-NEXT: arith.addi

    // CHECK: arith.constant 2
    // CHECK-NEXT: tensor_ext.rotate
    // CHECK-NEXT: tensor.extract_slice
    // CHECK-NEXT: arith.muli
    // CHECK-NEXT: arith.addi

    // CHECK: arith.constant 3
    // CHECK-NEXT: tensor_ext.rotate
    // CHECK-NEXT: tensor.extract_slice
    // CHECK-NEXT: arith.muli
    // CHECK-NEXT: arith.addi

    // CHECK: arith.constant 4
    // CHECK-NEXT: tensor_ext.rotate
    // CHECK-NEXT: tensor.extract_slice
    // CHECK-NEXT: arith.muli
    // CHECK-NEXT: arith.addi

    // CHECK: arith.constant 5
    // CHECK-NEXT: tensor_ext.rotate
    // CHECK-NEXT: tensor.extract_slice
    // CHECK-NEXT: arith.muli
    // CHECK-NEXT: arith.addi

    // CHECK: arith.constant 6
    // CHECK-NEXT: tensor_ext.rotate
    // CHECK-NEXT: tensor.extract_slice
    // CHECK-NEXT: arith.muli
    // CHECK-NEXT: arith.addi

    // CHECK: arith.constant 7
    // CHECK-NEXT: tensor_ext.rotate
    // CHECK-NEXT: tensor.extract_slice
    // CHECK-NEXT: arith.muli
    // CHECK-NEXT: arith.addi

    // CHECK: arith.constant 8
    // CHECK-NEXT: tensor_ext.rotate
    // CHECK-NEXT: tensor.extract_slice
    // CHECK-NEXT: arith.muli
    // CHECK-NEXT: arith.addi

    // CHECK: arith.constant 9
    // CHECK-NEXT: tensor_ext.rotate
    // CHECK-NEXT: tensor.extract_slice
    // CHECK-NEXT: arith.muli
    // CHECK-NEXT: arith.addi

    // CHECK: arith.constant 10
    // CHECK-NEXT: tensor_ext.rotate
    // CHECK-NEXT: tensor.extract_slice
    // CHECK-NEXT: arith.muli
    // CHECK-NEXT: arith.addi

    // CHECK: arith.constant 11
    // CHECK-NEXT: tensor_ext.rotate
    // CHECK-NEXT: tensor.extract_slice
    // CHECK-NEXT: arith.muli
    // CHECK-NEXT: arith.addi

    // CHECK: arith.constant 12
    // CHECK-NEXT: tensor_ext.rotate
    // CHECK-NEXT: tensor.extract_slice
    // CHECK-NEXT: arith.muli
    // CHECK-NEXT: arith.addi

    // CHECK: arith.constant 13
    // CHECK-NEXT: tensor_ext.rotate
    // CHECK-NEXT: tensor.extract_slice
    // CHECK-NEXT: arith.muli
    // CHECK-NEXT: arith.addi

    // CHECK: arith.constant 14
    // CHECK-NEXT: tensor_ext.rotate
    // CHECK-NEXT: tensor.extract_slice
    // CHECK-NEXT: arith.muli
    // CHECK-NEXT: arith.addi

    // CHECK: arith.constant 15
    // CHECK-NEXT: tensor_ext.rotate
    // CHECK-NEXT: tensor.extract_slice
    // CHECK-NEXT: arith.muli
    // CHECK-NEXT: arith.addi
    %3 = linalg.matvec {tensor_ext.layout = #vec_layout}
          ins(%enc_matrix, %input0 : tensor<16x16xi16>, tensor<16xi16>)
          outs(%enc_out : tensor<16xi16>) -> tensor<16xi16>

    // CHECK-NOT: tensor_ext.rotate
    secret.yield %3 : tensor<16xi16>
  } -> !secret.secret<tensor<16xi16>>
  // CHECK: return [[output]]
  return %0 : !secret.secret<tensor<16xi16>>
}

// -----

#input_vec_layout = #tensor_ext.layout<map = (d0) -> (d0 mod 16)>
#output_alignment = #tensor_ext.alignment<in = [4], out = [16], padding = [12], paddingValue = 0:i16>
#output_vec_layout = #tensor_ext.layout<map = (d0) -> (d0 mod 16), alignment = #output_alignment>
#diagonal = #tensor_ext.layout<map = (d0, d1) -> ((d1 - d0) mod 4, (d1 - (d1 - d0) mod 4) mod 16)>

// CHECK: [[row_major_indexing_map:#[^ ]*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: [[diagonal_layout:#[^ ]*]] = affine_map<(d0, d1) -> ((d1 - d0) mod 4, (d1 - (d1 - d0) mod 4) mod 16)>

// CHECK: @squat
// CHECK-SAME: [[arg0:%[^:]*]]: [[materialized_ty:!secret.secret<tensor<16xi16>>]]
func.func @squat(
    %arg0: !secret.secret<tensor<16xi16>> {tensor_ext.layout = #input_vec_layout}) ->
       (!secret.secret<tensor<4xi16>> {tensor_ext.layout = #output_vec_layout}) {
  // CHECK: [[cst:%[^ ]+]] = arith.constant dense<1> : tensor<4x16xi16>
  %cst = arith.constant dense<1> : tensor<4x16xi16>
  // CHECK: [[loop_init:%[^ ]+]] = arith.constant dense<0> : tensor<4xi16>
  %out = arith.constant dense<0> : tensor<4xi16>

  // CHECK: [[output:%[^ ]+]] = secret.generic
  // CHECK-SAME: ([[arg0]]
  %0 = secret.generic(%arg0 : !secret.secret<tensor<16xi16>>)
                      attrs = {
                        __argattrs = [{tensor_ext.layout = #input_vec_layout}],
                        __resattrs = [{tensor_ext.layout = #output_vec_layout}]
                      } {
  // CHECK: ^body([[clear_arg0:%[^ ]+]]: tensor<16xi16>):
  ^body(%input0: tensor<16xi16>):
    %enc_out = tensor_ext.assign_layout %out {layout = #output_vec_layout, tensor_ext.layout = #output_vec_layout} : tensor<4xi16>

    // Apply the diagonal encoding
    // CHECK: [[enc_matrix_out:%[^ ]+]] = arith.constant dense<0> : tensor<4x16xi16>
    // CHECK: [[enc_matrix:%[^ ]+]] = linalg.generic
    // CHECK-SAME: indexing_maps = [
    // CHECK-SAME: [[row_major_indexing_map]],
    // CHECK-SAME: [[diagonal_layout]]]
    // CHECK-SAME: iterator_types = [
    // CHECK-SAME: "parallel", "parallel"]}
    // CHECK-NEXT: ^bb0([[in:%[^ ]+]]: i16, [[out:%[^ ]+]]: i16):
    // CHECK-NEXT: linalg.yield [[in]]
    %enc_matrix = tensor_ext.assign_layout %cst {layout = #diagonal, tensor_ext.layout = #diagonal} : tensor<4x16xi16>

    // Now the Halevi-Shoup kernel
    // 4 iterations, unrolled
    // CHECK: ensor_ext.rotate
    // CHECK-NEXT: tensor.extract_slice
    // CHECK-NEXT: arith.muli
    // CHECK-NEXT: arith.addi

    // CHECK: tensor_ext.rotate
    // CHECK-NEXT: tensor.extract_slice
    // CHECK-NEXT: arith.muli
    // CHECK-NEXT: arith.addi

    // CHECK: tensor_ext.rotate
    // CHECK-NEXT: tensor.extract_slice
    // CHECK-NEXT: arith.muli
    // CHECK-NEXT: arith.addi

    // CHECK: tensor_ext.rotate
    // CHECK-NEXT: tensor.extract_slice
    // CHECK-NEXT: arith.muli
    // CHECK-NEXT: [[loop_output:%[^ ]*]] = arith.addi
    %3 = linalg.matvec {tensor_ext.layout = #output_vec_layout}
          ins(%enc_matrix, %input0 : tensor<4x16xi16>, tensor<16xi16>)
          outs(%enc_out : tensor<4xi16>) -> tensor<4xi16>

    // Now the partial reduction step
    // CHECK-NEXT: [[c8:%[^ ]*]] = arith.constant 8
    // CHECK-NEXT: [[rotated_by_8:%[^ ]*]] = tensor_ext.rotate [[loop_output]], [[c8]]
    // CHECK-NEXT: [[summed1:%[^ ]*]] = arith.addi [[loop_output]], [[rotated_by_8]]
    // CHECK-NEXT: [[c4:%[^ ]*]] = arith.constant 4
    // CHECK-NEXT: [[rotated_by_4:%[^ ]*]] = tensor_ext.rotate [[summed1]], [[c4]]
    // CHECK-NEXT: [[summed2:%[^ ]*]] = arith.addi [[summed1]], [[rotated_by_4]]

    // Now the plaintext mask
    // CHECK-NEXT: [[mask_zeros:%[^ ]*]] = arith.constant dense<0> : tensor<16xi16>
    // CHECK-NEXT: [[mask_ones:%[^ ]*]] = arith.constant dense<1> : tensor<4xi16>
    // CHECK-NEXT: [[mask:%[^ ]*]] = tensor.insert_slice [[mask_ones]] into [[mask_zeros]][0] [4] [1]
    // CHECK-NEXT: [[yielded_val:%[^ ]*]] = arith.muli [[summed2]], [[mask]]

    // CHECK: secret.yield [[yielded_val]]
    secret.yield %3 : tensor<4xi16>
  } -> !secret.secret<tensor<4xi16>>
  // CHECK: return [[output]]
  return %0 : !secret.secret<tensor<4xi16>>
}

// -----

// Test f32 padding values

#input_vec_layout = #tensor_ext.layout<map = (d0) -> (d0 mod 16)>
#output_alignment = #tensor_ext.alignment<in = [4], out = [16], padding = [12], paddingValue = 0.0:f32>
#output_vec_layout = #tensor_ext.layout<map = (d0) -> (d0 mod 16), alignment = #output_alignment>
#diagonal = #tensor_ext.layout<map = (d0, d1) -> ((d1 - d0) mod 4, (d1 - (d1 - d0) mod 4) mod 16)>

// CHECK: @f32_padding
// CHECK-SAME: [[arg0:%[^:]*]]: [[materialized_ty:!secret.secret<tensor<16xf32>>]]
// CHECK: [[cst:%[^ ]+]] = arith.constant dense<1.000000e+00> : tensor<4x16xf32>
// CHECK: [[loop_init:%[^ ]+]] = arith.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK: [[enc_matrix_out:%[^ ]+]] = arith.constant dense<0.000000e+00> : tensor<4x16xf32>
// CHECK: [[mask_zeros:%[^ ]*]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// CHECK: [[mask_ones:%[^ ]*]] = arith.constant dense<1.000000e+00> : tensor<4xf32>
func.func @f32_padding(
    %arg0: !secret.secret<tensor<16xf32>> {tensor_ext.layout = #input_vec_layout}) ->
       (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #output_vec_layout}) {
  %cst = arith.constant dense<1.0> : tensor<4x16xf32>
  %out = arith.constant dense<0.0> : tensor<4xf32>

  %0 = secret.generic(%arg0 : !secret.secret<tensor<16xf32>>)
                      attrs = {
                        __argattrs = [{tensor_ext.layout = #input_vec_layout}],
                        __resattrs = [{tensor_ext.layout = #output_vec_layout}]
                      } {
  ^body(%input0: tensor<16xf32>):
    %enc_out = tensor_ext.assign_layout %out {layout = #output_vec_layout, tensor_ext.layout = #output_vec_layout} : tensor<4xf32>

    %enc_matrix = tensor_ext.assign_layout %cst {layout = #diagonal, tensor_ext.layout = #diagonal} : tensor<4x16xf32>
    %3 = linalg.matvec {tensor_ext.layout = #output_vec_layout}
          ins(%enc_matrix, %input0 : tensor<4x16xf32>, tensor<16xf32>)
          outs(%enc_out : tensor<4xf32>) -> tensor<4xf32>

    secret.yield %3 : tensor<4xf32>
  } -> !secret.secret<tensor<4xf32>>
  return %0 : !secret.secret<tensor<4xf32>>
}
