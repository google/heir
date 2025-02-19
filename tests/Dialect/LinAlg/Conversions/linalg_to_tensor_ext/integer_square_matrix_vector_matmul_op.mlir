// RUN: heir-opt %s --linalg-to-tensor-ext=tiling-size=4 --canonicalize | FileCheck %s

// CHECK:      func.func @test_integer_vector_square_matrix_matmul(%[[ARG:.*]]: !secret.secret<tensor<4x1xi16>>)
// CHECK-DAG:  %[[ONE:.*]] = arith.constant 1 : index
// CHECK:      %[[DIAGONALIZED_MATRIX:.*]] = arith.constant dense
// CHECK-SAME{LITERAL}: <[[1, 2, 3, 4], [6, 7, 8, 5], [11, 12, 9, 10], [16, 13, 14, 15]]> : tensor<4x4xi16>
// CHECK:      %[[BIAS:.*]] = arith.constant dense
// CHECK-SAME{LITERAL}: <[[17], [18], [19], [20]]> : tensor<4x1xi16>
// CHECK:      %[[FIRST_SLICE:.*]] = tensor.extract_slice %[[DIAGONALIZED_MATRIX]][0, 0] [4, 1] [1, 1]
// CHECK:      %[[OUT:.*]] = secret.generic ins(%[[ARG]] : !secret.secret<tensor<4x1xi16>>)
// CHECK:      ^body(%[[ARG_CONVERTED:.*]]: tensor<4x1xi16>):
// CHECK:        %[[FIRST_MUL:.*]] = arith.muli %[[ARG_CONVERTED]], %[[FIRST_SLICE]]
// CHECK:        %[[FIRST_SUM:.*]] = arith.addi %[[FIRST_MUL]], %[[BIAS]]
// CHECK:        %[[FOR_LOOP_OUT:.*]]:2 = affine.for %[[I:.*]] = 1 to 4 iter_args(%[[RUNNING_SUM:.*]] = %[[FIRST_SUM]], %[[ROTATED_VEC:.*]] = %[[ARG_CONVERTED]])
// CHECK-DAG:      %[[UPDATED_ROTATED_VEC:.*]] = tensor_ext.rotate %[[ROTATED_VEC]], %[[ONE]]
// CHECK-DAG:      %[[SLICE:.*]] = tensor.extract_slice %[[DIAGONALIZED_MATRIX]][0, %[[I]]] [4, 1] [1, 1]
// CHECK:          %[[MUL:.*]] = arith.muli %[[UPDATED_ROTATED_VEC]], %[[SLICE]]
// CHECK:          %[[UPDATED_SUM:.*]] = arith.addi %[[RUNNING_SUM]], %[[MUL]]
// CHECK:          affine.yield %[[UPDATED_SUM]], %[[UPDATED_ROTATED_VEC]]
// CHECK:      secret.yield %[[FOR_LOOP_OUT]]#0
// CHECK:      return %[[OUT]]
module {
func.func @test_integer_vector_square_matrix_matmul(%vec : !secret.secret<tensor<4x1xi16>>) -> !secret.secret<tensor<4x1xi16>> {
  %matrix = arith.constant dense<[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]> : tensor<4x4xi16>
  %bias = arith.constant dense<[[17], [18], [19], [20]]> : tensor<4x1xi16>
  %out = secret.generic ins (%vec : !secret.secret<tensor<4x1xi16>>) {
  ^body(%converted_vec: tensor<4x1xi16>):
    %0 = linalg.matmul ins(%matrix, %converted_vec : tensor<4x4xi16>, tensor<4x1xi16>) outs(%bias : tensor<4x1xi16>) -> tensor<4x1xi16>
    secret.yield %0 : tensor<4x1xi16>
  } -> !secret.secret<tensor<4x1xi16>>
  return %out : !secret.secret<tensor<4x1xi16>>
}
}
