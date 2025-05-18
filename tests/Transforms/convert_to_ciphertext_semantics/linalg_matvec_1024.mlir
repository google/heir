// RUN: heir-opt %s --convert-to-ciphertext-semantics=ciphertext-size=1024 | FileCheck %s

// This test checks an edge case of the matvec kernel where the squat kernel
// rotate-and-reduce was being applied to square tensors incorrectly.

#vec_layout = #tensor_ext.layout<map = (d0) -> (d0 mod 1024)>
#diagonal = #tensor_ext.layout<map = (d0, d1) -> ((d1 - d0) mod 16, (d1 - (d1 - d0) mod 16) mod 1024)>

// CHECK-DAG: [[row_major_indexing_map:#[^ ]*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: [[diagonal_layout:#[^ ]*]] = affine_map<(d0, d1) -> ((d1 - d0) mod 16, (d1 - (d1 - d0) mod 16) mod 1024)>
// CHECK-DAG: [[layout_rm_1d:#[^ ]*]] = #tensor_ext.layout<map = (d0) -> (d0 mod 1024)>
// CHECK-DAG: [[orig_type:#[^ ]*]] = #tensor_ext.original_type<originalType = tensor<16xi16>, layout = [[layout_rm_1d]]>

// CHECK: @matvec_constant_matrix
// CHECK-SAME: [[arg0:%[^:]*]]: [[materialized_ty:!secret.secret<tensor<1024xi16>>]]
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
  // CHECK: ^body([[clear_arg0:%[^ ]+]]: tensor<1024xi16>):
  ^body(%input0: tensor<16xi16>):
    // Apply the row-major encoding
    // CHECK: [[enc_vec_out:%[^ ]+]] = arith.constant dense<0> : tensor<1024xi16>
    // CHECK: [[enc_vec:%[^ ]+]] = linalg.generic
    // CHECK-SAME: [[loop_init]]
    %enc_out = tensor_ext.assign_layout %out {layout = #vec_layout, tensor_ext.layout = #vec_layout} : tensor<16xi16>

    // Apply the diagonal encoding
    // CHECK: [[enc_matrix_out:%[^ ]+]] = arith.constant dense<0> : tensor<16x1024xi16>
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
