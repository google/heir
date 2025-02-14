// RUN: heir-opt %s --linalg-to-tensor-ext=tiling-size=4 --canonicalize | FileCheck %s

// CHECK:      func.func @test_float_vector_small_matrix_matmul(%[[ARG:.*]]: !secret.secret<tensor<1x4xf32>>)
// CHECK-DAG:   %[[TWO:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[BIAS:.*]] = arith.constant dense<5.{{0*}}e+00> : tensor<1x4xf32>
// CHECK-DAG:   %[[DIAGONALIZED_MATRIX:.*]] = arith.constant dense
// CHECK-SAME{LITERAL}: <[[
// CHECK-SAME: 1.{{0*}}e+00, 2.{{0*}}e+00, 3.{{0*}}e+00, 4.{{0*}}e+00], [2.{{0*}}e+00, 3.{{0*}}e+00, 4.{{0*}}e+00, 1.{{0*}}e+00], [3.{{0*}}e+00, 4.{{0*}}e+00, 1.{{0*}}e+00, 2.{{0*}}e+00], [4.{{0*}}e+00, 1.{{0*}}e+00, 2.{{0*}}e+00, 3.{{0*}}e+00
// CHECK-SAME{LITERAL}: ]]>
// CHECK-DAG:   %[[SLICE:.*]] = tensor.extract_slice %[[DIAGONALIZED_MATRIX]][0, 0] [1, 4] [1, 1]
// CHECK:       %[[OUT:.*]] = secret.generic ins(%[[ARG]] : !secret.secret<tensor<1x4xf32>>)
// CHECK:       ^body(%[[ARG_CONVERTED:.*]]: tensor<1x4xf32>):
// CHECK:         %[[MUL:.*]] = arith.mulf %[[ARG_CONVERTED]], %[[SLICE]]
// CHECK:         %[[SUM:.*]] = arith.addf %[[MUL]], %[[BIAS]]
// CHECK:         %[[ROTATE1:.*]] = tensor_ext.rotate %[[SUM]], %[[TWO]]
// CHECK:         %[[ROTATE_AND_SUM_1:.*]] = arith.addf %[[SUM]], %[[ROTATE1]]
// CHECK:         %[[ROTATE2:.*]] = tensor_ext.rotate %[[ROTATE_AND_SUM_1]], %[[ONE]]
// CHECK:         %[[FINAL_SUM:.*]] = arith.addf %[[ROTATE_AND_SUM_1]], %[[ROTATE2]]
// CHECK:         secret.yield %[[FINAL_SUM]]
// CHECK:       return %[[OUT]]
module {
func.func @test_float_vector_small_matrix_matmul(%vec : !secret.secret<tensor<1x4xf32>>) -> !secret.secret<tensor<1x1xf32>> {
  %matrix = arith.constant dense<[[1.0], [2.0], [3.0], [4.0]]> : tensor<4x1xf32>
  %bias = arith.constant dense<[[5.0]]> : tensor<1x1xf32>
  %out = secret.generic ins (%vec : !secret.secret<tensor<1x4xf32>>) {
  ^bb0(%converted_vec: tensor<1x4xf32>):
    %0 = linalg.matmul ins(%converted_vec, %matrix : tensor<1x4xf32>, tensor<4x1xf32>) outs(%bias : tensor<1x1xf32>) -> tensor<1x1xf32>
    secret.yield %0 : tensor<1x1xf32>
  } -> !secret.secret<tensor<1x1xf32>>
  return %out : !secret.secret<tensor<1x1xf32>>
}
}
