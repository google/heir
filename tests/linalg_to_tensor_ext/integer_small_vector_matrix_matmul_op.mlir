// RUN: heir-opt %s --linalg-to-tensor-ext=tiling-size=4 --canonicalize | FileCheck %s

// CHECK:      func.func @test_integer_small_vector_rect_matrix_matmul(%[[ARG:.*]]: !secret.secret<tensor<1x4xi16>>)
// CHECK-DAG:   %[[DIAGONALIZED_MATRIX:.*]] = arith.constant dense
// CHECK-SAME{LITERAL}: <[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]> : tensor<4x4xi16>
// CHECK-DAG:   %[[BIAS:.*]] = arith.constant dense
// CHECK-SAME{LITERAL}: <[[5, 6, 7, 8]]> : tensor<1x4xi16>
// CHECK-DAG:   %[[SLICE:.*]] = tensor.extract_slice %[[DIAGONALIZED_MATRIX]][0, 0] [1, 4] [1, 1]
// CHECK:       %[[OUT:.*]] = secret.generic ins(%[[ARG]] : !secret.secret<tensor<1x4xi16>>)
// CHECK:       ^bb0(%[[ARG_CONVERTED:.*]]: tensor<1x4xi16>):
// CHECK:         %[[MUL:.*]] = arith.muli %[[ARG_CONVERTED]], %[[SLICE]]
// CHECK:         %[[FINAL_SUM:.*]] = arith.addi %[[MUL]], %[[BIAS]]
// CHECK:         secret.yield %[[FINAL_SUM]]
// CHECK:       return %[[OUT]]
module {
func.func @test_integer_small_vector_rect_matrix_matmul(%vec : !secret.secret<tensor<1x1xi16>>) -> !secret.secret<tensor<1x4xi16>> {
  %matrix = arith.constant dense<[[1, 2, 3, 4]]> : tensor<1x4xi16>
  %bias = arith.constant dense<[[5, 6, 7, 8]]> : tensor<1x4xi16>
  %out = secret.generic ins (%vec : !secret.secret<tensor<1x1xi16>>) {
  ^bb0(%converted_vec: tensor<1x1xi16>):
    %0 = linalg.matmul ins(%converted_vec, %matrix : tensor<1x1xi16>, tensor<1x4xi16>) outs(%bias : tensor<1x4xi16>) -> tensor<1x4xi16>
    secret.yield %0 : tensor<1x4xi16>
  } -> !secret.secret<tensor<1x4xi16>>
  return %out : !secret.secret<tensor<1x4xi16>>
}
}
