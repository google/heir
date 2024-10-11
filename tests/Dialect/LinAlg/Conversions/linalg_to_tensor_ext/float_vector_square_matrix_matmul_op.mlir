// RUN: heir-opt %s --linalg-to-tensor-ext | FileCheck %s

// CHECK:      func.func @test_float_vector_square_matrix_linalg_to_arith(%[[ARG:.*]]: !secret.secret<tensor<1x4xf16>>)
// CHECK-DAG:      %[[ONE:.*]] = arith.constant 1 : index
// CHECK:      %[[DIAGONALIZED_MATRIX:.*]] = arith.constant dense
// CHECK-SAME{LITERAL}: <[[
// CHECK-SAME: 1.{{0*}}e+00, 6.{{0*}}e+00, 1.1{{0*}}e+01, 1.6{{0*}}e+01], [5.{{0*}}e+00, 1.{{0*}}e+01, 1.5{{0*}}e+01, 4.{{0*}}e+00], [9.{{0*}}e+00, 1.4{{0*}}e+01, 3.{{0*}}e+00, 8.{{0*}}e+00], [1.3{{0*}}e+01, 2.{{0*}}e+00, 7.{{0*}}e+00, 1.2{{0*}}e+01
// CHECK-SAME{LITERAL}: ]]>
// CHECK:     %[[BIAS:.*]] = arith.constant dense
// CHECK-SAME{LITERAL}: <[[
// CHECK-SAME: 1.7{{0*}}e+01, 1.8{{0*}}e+01, 1.9{{0*}}e+01, 2.{{0*}}e+01
// CHECK-SAME{LITERAL}: ]]>
// CHECK:      %[[OUT:.*]] = secret.generic ins(%[[ARG]] : !secret.secret<tensor<1x4xf16>>)
// CHECK:      ^bb0(%[[ARG_CONVERTED:.*]]: tensor<1x4xf16>):
// CHECK:        %[[FOR_LOOP_OUT:.*]]:2 = affine.for %[[I:.*]] = 0 to 3 iter_args(%[[RUNNING_SUM:.*]] = %[[BIAS]], %[[ROTATED_VEC:.*]] = %[[ARG_CONVERTED]])
// CHECK:        %[[SLICE:.*]] = tensor.extract_slice %[[DIAGONALIZED_MATRIX]][%[[I]], 0] [1, 4] [1, 1]
// CHECK:        %[[MUL:.*]] = arith.mulf %[[ROTATED_VEC]], %[[SLICE]]
// CHECK:        %[[UPDATED_SUM:.*]] = arith.addf %[[RUNNING_SUM]], %[[MUL]]
// CHECK:        %[[UPDATED_ROTATED_VEC:.*]] = tensor_ext.rotate %[[ROTATED_VEC]], %[[ONE]]
// CHECK:        affine.yield %[[UPDATED_SUM]], %[[UPDATED_ROTATED_VEC]]
// CHECK:      %[[LAST_SLICE:.*]] = tensor.extract_slice %[[DIAGONALIZED_MATRIX]][3, 0] [1, 4] [1, 1]
// CHECK:      %[[LAST_MUL:.*]] = arith.mulf %[[FOR_LOOP_OUT]]#1, %[[LAST_SLICE]]
// CHECK:      %[[FINAL_SUM:.*]] = arith.addf %[[FOR_LOOP_OUT]]#0, %[[LAST_MUL]]
// CHECK:      secret.yield %[[FINAL_SUM]]
// CHECK:      return %[[OUT]]
module {
func.func @test_float_vector_square_matrix_linalg_to_arith(%vec : !secret.secret<tensor<1x4xf16>>) -> !secret.secret<tensor<1x4xf16>> {
  %matrix = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]> : tensor<4x4xf16>
  %bias = arith.constant dense<[[17.0, 18.0, 19.0, 20.0]]> : tensor<1x4xf16>
  %out = secret.generic ins (%vec : !secret.secret<tensor<1x4xf16>>) {
  ^bb0(%converted_vec: tensor<1x4xf16>):
    %0 = linalg.matmul ins(%converted_vec, %matrix : tensor<1x4xf16>, tensor<4x4xf16>) outs(%bias : tensor<1x4xf16>) -> tensor<1x4xf16>
    secret.yield %0 : tensor<1x4xf16>
  } -> !secret.secret<tensor<1x4xf16>>
  return %out : !secret.secret<tensor<1x4xf16>>
}
}
