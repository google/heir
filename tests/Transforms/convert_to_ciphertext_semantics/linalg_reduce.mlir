// RUN: heir-opt %s --convert-to-ciphertext-semantics=ciphertext-size=16 | FileCheck %s

#row_major_matrix = #tensor_ext.layout<map = (d0, d1) -> (d0 * 4 + d1)>
#row_major_vec_align = #tensor_ext.alignment<in = [4], out = [16]>
#row_major_vec = #tensor_ext.layout<map = (d0) -> (d0), alignment=#row_major_vec_align>
#col_major_vec_align = #tensor_ext.alignment<in = [4], out = [16], padding = [12], paddingValue = 0:i16>
#col_major_vec = #tensor_ext.layout<map = (d0) -> (d0 * 4), alignment=#col_major_vec_align>


// CHECK: [[align:#[^ ]*]] = #tensor_ext.alignment<in = [4], out = [16]>
// CHECK: [[layout:#[^ ]*]] = #tensor_ext.layout<map = (d0) -> (d0), alignment = [[align]]>
// CHECK: [[orig_ty:#[^ ]*]] = #tensor_ext.original_type<originalType = tensor<4xi16>, layout = [[layout]]>

// CHECK: @convert_linalg_reduce
// CHECK-SAME: [[arg0:%[^:]*]]: [[materialized_ty:!secret.secret<tensor<16xi16>>]]
// CHECK-SAME: tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<4x4xi16>, layout = <map = (d0, d1) -> (d0 * 4 + d1)>>}
// CHECK-SAME: [[arg1:%[^:]*]]: [[materialized_ty]]
// CHECK-SAME: tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<4x4xi16>, layout = <map = (d0, d1) -> (d0 * 4 + d1)>>
// CHECK-SAME: -> ([[materialized_ty]] {tensor_ext.original_type = [[orig_ty]]})
func.func @convert_linalg_reduce(
    %arg0: !secret.secret<tensor<4x4xi16>> {tensor_ext.layout = #row_major_matrix},
    %arg1: !secret.secret<tensor<4x4xi16>> {tensor_ext.layout = #row_major_matrix}) ->
       (!secret.secret<tensor<4xi16>> {tensor_ext.layout = #row_major_vec}) {
  // CHECK: [[cst:%[^ ]+]] = arith.constant dense<0> : tensor<4xi16>
  %cst = arith.constant dense<0> : tensor<4xi16>

  // lifted the second tensor_ext.assign_layout

  // CHECK: secret.generic ins(
  // CHECK-SAME: [[arg0]], [[arg1]]
  // CHECK-NEXT: ^body(
  // CHECK-SAME: [[pt_arg0:%[^ ]*]]:
  // CHECK-SAME: [[pt_arg1:%[^ ]*]]:
  %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<4x4xi16>>, !secret.secret<tensor<4x4xi16>>)
                      attrs = {
                        __argattrs = [{tensor_ext.layout = #row_major_matrix}, {tensor_ext.layout = #row_major_matrix}],
                        __resattrs = [{tensor_ext.layout = #row_major_vec}]
                      } {
  ^body(%input0: tensor<4x4xi16>, %input1: tensor<4x4xi16>):
    // CHECK-NEXT: tensor.empty() : tensor<16xi16>
    // CHECK-NEXT: tensor.insert_slice [[cst]] into
    // CHECK-SAME: [0] [4] [1]
    // CHECK-NEXT: tensor.insert_slice [[cst]] into
    // CHECK-SAME: [4] [4] [1]
    // CHECK-NEXT: tensor.insert_slice [[cst]] into
    // CHECK-SAME: [8] [4] [1]
    // CHECK-NEXT: [[pt_cst1:[^ ]*]] = tensor.insert_slice [[cst]] into
    // CHECK-SAME: [12] [4] [1]
    %1 = tensor_ext.assign_layout %cst {layout = #row_major_vec, tensor_ext.layout = #row_major_vec} : tensor<4xi16>

    // First reduction
    // CHECK:  [[rotate1:%[^ ]*]] = tensor_ext.permute [[pt_arg0]] {permutation = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]>
    // CHECK:  [[sum1:%[^ ]*]] = arith.addi [[pt_cst1]], [[rotate1]]
    // CHECK:  [[rotate2:%[^ ]*]] = tensor_ext.permute [[pt_arg0]] {permutation = dense<[12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]>
    // CHECK:  [[sum2:%[^ ]*]] = arith.addi [[sum1]], [[rotate2]] : tensor<16xi16>
    // CHECK:  [[rotate3:%[^ ]*]] = tensor_ext.permute [[pt_arg0]] {permutation = dense<[8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7]>
    // CHECK:  [[sum3:%[^ ]*]] = arith.addi [[sum2]], [[rotate3]] : tensor<16xi16>
    // CHECK:  [[rotate4:%[^ ]*]] = tensor_ext.permute [[pt_arg0]] {permutation = dense<[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3]>
    // CHECK:  [[sum4:%[^ ]*]] = arith.addi [[sum3]], [[rotate4]] : tensor<16xi16>
    %reduced = linalg.reduce { arith.addi {overflowFlags = #arith.overflow<none>} }
      ins(%input0 : tensor<4x4xi16>)
      outs(%1 : tensor<4xi16>)
      dimensions = [0]  {tensor_ext.layout = #row_major_vec}

    // CHECK-NEXT: tensor.empty() : tensor<16xi16>
    // CHECK-NEXT: tensor.insert_slice [[cst]] into
    // CHECK-SAME: [0] [4] [1]
    // CHECK-NEXT: tensor.insert_slice [[cst]] into
    // CHECK-SAME: [4] [4] [1]
    // CHECK-NEXT: tensor.insert_slice [[cst]] into
    // CHECK-SAME: [8] [4] [1]
    // CHECK-NEXT: [[pt_cst2:[^ ]*]] = tensor.insert_slice [[cst]] into
    // CHECK-SAME: [12] [4] [1]
    %2 = tensor_ext.assign_layout %cst {layout = #row_major_vec, tensor_ext.layout = #row_major_vec} : tensor<4xi16>
    // CHECK:  [[converted1:%[^ ]*]] = tensor_ext.permute [[pt_cst2]] {permutation = dense<[0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]>
    %3 = tensor_ext.convert_layout %2 {from_layout = #row_major_vec, tensor_ext.layout = #col_major_vec, to_layout = #col_major_vec} : tensor<4xi16>

    // second reduction
    // CHECK:  [[rotate1_2:%[^ ]*]] = tensor_ext.permute [[pt_arg1]] {permutation = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]>
    // CHECK:  [[sum1_2:%[^ ]*]] = arith.addi [[converted1]], [[rotate1_2]]
    // CHECK:  [[rotate2_2:%[^ ]*]] = tensor_ext.permute [[pt_arg1]] {permutation = dense<[15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]>
    // CHECK:  [[sum2_2:%[^ ]*]] = arith.addi [[sum1_2]], [[rotate2_2]] : tensor<16xi16>
    // CHECK:  [[rotate3_2:%[^ ]*]] = tensor_ext.permute [[pt_arg1]] {permutation = dense<[14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]>
    // CHECK:  [[sum3_2:%[^ ]*]] = arith.addi [[sum2_2]], [[rotate3_2]] : tensor<16xi16>
    // CHECK:  [[rotate4_2:%[^ ]*]] = tensor_ext.permute [[pt_arg1]] {permutation = dense<[13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]>
    // CHECK:  [[sum4_2:%[^ ]*]] = arith.addi [[sum3_2]], [[rotate4_2]] : tensor<16xi16>
    %reduced_1 = linalg.reduce { arith.addi {overflowFlags = #arith.overflow<none>} }
      ins(%input1 : tensor<4x4xi16>)
      outs(%3 : tensor<4xi16>)
      dimensions = [1]  {tensor_ext.layout = #col_major_vec}

    // CHECK: [[converted2:%[^ ]*]] = tensor_ext.permute [[sum4_2]] {permutation = dense<[0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]>
    %4 = tensor_ext.convert_layout %reduced_1 {from_layout = #col_major_vec, tensor_ext.layout = #row_major_vec, to_layout = #row_major_vec} : tensor<4xi16>
    // CHECK: arith.addi [[sum4]], [[converted2]]
    %5 = arith.addi %reduced, %4 {tensor_ext.layout = #row_major_vec} : tensor<4xi16>
    secret.yield %5 : tensor<4xi16>
  } -> !secret.secret<tensor<4xi16>>
  return %0 : !secret.secret<tensor<4xi16>>
}

func.func @reduce_mulop(%arg0: !secret.secret<tensor<4x4xi16>> {tensor_ext.layout = #row_major_matrix}) ->
       (!secret.secret<tensor<4xi16>> {tensor_ext.layout = #row_major_vec}) {
  %cst = arith.constant dense<0> : tensor<4xi16>
  %0 = secret.generic ins(%arg0 : !secret.secret<tensor<4x4xi16>>)
                      attrs = {
                        __argattrs = [{tensor_ext.layout = #row_major_matrix}],
                        __resattrs = [{tensor_ext.layout = #row_major_vec}]
                      } {
  ^body(%input0: tensor<4x4xi16>):
    %1 = tensor_ext.assign_layout %cst {layout = #row_major_vec, tensor_ext.layout = #row_major_vec} : tensor<4xi16>

    // CHECK-COUNT-4: arith.muli
    %reduced = linalg.reduce { arith.muli }
      ins(%input0 : tensor<4x4xi16>)
      outs(%1 : tensor<4xi16>)
      dimensions = [0]  {tensor_ext.layout = #row_major_vec}

    secret.yield %reduced : tensor<4xi16>
  } -> !secret.secret<tensor<4xi16>>
  return %0 : !secret.secret<tensor<4xi16>>
}
