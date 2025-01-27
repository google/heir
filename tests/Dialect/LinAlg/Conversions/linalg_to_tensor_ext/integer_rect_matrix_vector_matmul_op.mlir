// RUN: heir-opt %s --linalg-to-tensor-ext=tiling-size=4 --canonicalize | FileCheck %s

// CHECK:       func.func @test_integer_rect_matrix_vector_matmul(%[[ARG:.*]]: !secret.secret<tensor<4x1xi16>>)
// CHECK-DAG:   %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[TWO:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[BIAS:.*]] = arith.constant dense
// CHECK-SAME{LITERAL}: <[[17], [18], [17], [18]]> : tensor<4x1xi16>
// CHECK-DAG:   %[[DIAGONALIZED_MATRIX:.*]] = arith.constant dense
// CHECK-SAME{LITERAL}: <[[1, 2, 3, 4], [6, 7, 8, 5], [3, 4, 1, 2], [8, 5, 6, 7]]> : tensor<4x4xi16>
// CHECK-DAG:   %[[LAST_SLICE:.*]] = tensor.extract_slice %[[DIAGONALIZED_MATRIX]][0, 1] [4, 1] [1, 1]
// CHECK:       %[[OUT:.*]] = secret.generic ins(%[[ARG]] : !secret.secret<tensor<4x1xi16>>)
// CHECK:       ^body(%[[ARG_CONVERTED:.*]]: tensor<4x1xi16>):
// CHECK:         %[[FOR_LOOP_OUT:.*]]:2 = affine.for %[[I:.*]] = 0 to 1 iter_args(%[[RUNNING_SUM:.*]] = %[[BIAS]], %[[ROTATED_VEC:.*]] = %[[ARG_CONVERTED]])
// CHECK:         %[[SLICE:.*]] = tensor.extract_slice %[[DIAGONALIZED_MATRIX]][0, %[[I]]] [4, 1] [1, 1]
// CHECK:         %[[MUL:.*]] = arith.muli %[[ROTATED_VEC]], %[[SLICE]]
// CHECK:         %[[UPDATED_SUM:.*]] = arith.addi %[[RUNNING_SUM]], %[[MUL]]
// CHECK:         %[[UPDATED_ROTATED_VEC:.*]] = tensor_ext.rotate %[[ROTATED_VEC]], %[[ONE]]
// CHECK:         affine.yield %[[UPDATED_SUM]], %[[UPDATED_ROTATED_VEC]]
// CHECK:       %[[LAST_MUL:.*]] = arith.muli %[[FOR_LOOP_OUT]]#1, %[[LAST_SLICE]]
// CHECK:       %[[BEFORE_ROTATE_AND_SUM:.*]] = arith.addi %[[FOR_LOOP_OUT]]#0, %[[LAST_MUL]]
// CHECK:       %[[ROTATED_SUM:.*]] = tensor_ext.rotate %[[BEFORE_ROTATE_AND_SUM]], %[[TWO]]
// CHECK:       %[[FINAL_SUM:.*]] = arith.addi %[[BEFORE_ROTATE_AND_SUM]], %[[ROTATED_SUM]]
// CHECK:       secret.yield %[[FINAL_SUM]]
// CHECK:       return %[[OUT]]
module {
func.func @test_integer_rect_matrix_vector_matmul(%vec : !secret.secret<tensor<4x1xi16>>) -> !secret.secret<tensor<2x1xi16>> {
  %matrix = arith.constant dense<[[1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi16>
  %bias = arith.constant dense<[[17], [18]]> : tensor<2x1xi16>
  %out = secret.generic ins (%vec : !secret.secret<tensor<4x1xi16>>) {
  ^bb0(%converted_vec: tensor<4x1xi16>):
    %0 = linalg.matmul ins(%matrix, %converted_vec : tensor<2x4xi16>, tensor<4x1xi16>) outs(%bias : tensor<2x1xi16>) -> tensor<2x1xi16>
    secret.yield %0 : tensor<2x1xi16>
  } -> !secret.secret<tensor<2x1xi16>>
  return %out : !secret.secret<tensor<2x1xi16>>
}
}
