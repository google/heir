// RUN: heir-opt %s --tosa-to-secret-arith | FileCheck %s

// CHECK:      func.func @test_tosa_sigmoid_to_secret_arith(%[[ARG:.*]]: !secret.secret<tensor<1x16xf32>>)
// CHECK-DAG:   %[[COEFF_0:.*]] = arith.constant dense<5.{{0*}}e-01> : tensor<1x16xf32>
// CHECK-DAG:   %[[COEFF_1:.*]] = arith.constant dense<1.97{{0*}}e-01> : tensor<1x16xf32>
// CHECK-DAG:   %[[COEFF_3:.*]] = arith.constant dense<-4.{{0*}}e-03> : tensor<1x16xf32>
// CHECK:       %[[RET:.*]] = secret.generic ins(%[[ARG]] : !secret.secret<tensor<1x16xf32>>)
// CHECK-NEXT:  ^bb0(%[[CONVERTED_ARG:.*]]: tensor<1x16xf32>):
// CHECK:         %[[COEFF_MUL_DEGREE_1:.*]] = arith.mulf %[[CONVERTED_ARG]], %[[COEFF_1]]
// CHECK:         %[[DEGREE_2:.*]] = arith.mulf %[[CONVERTED_ARG]], %[[CONVERTED_ARG]]
// CHECK:         %[[DEGREE_3:.*]] = arith.mulf %[[DEGREE_2]], %[[CONVERTED_ARG]]
// CHECK:         %[[COEFF_MUL_DEGREE_3:.*]] = arith.mulf %[[DEGREE_3]], %[[COEFF_3]]
// CHECK:         %[[SUM_1:.*]] = arith.addf %[[COEFF_MUL_DEGREE_1]], %[[COEFF_0]]
// CHECK:         %[[TOTAL_SUM:.*]] = arith.addf %[[SUM_1]], %[[COEFF_MUL_DEGREE_3]]
// CHECK:         secret.yield %[[TOTAL_SUM]] : tensor<1x16xf32>
// CHECK:       return %[[RET]] : !secret.secret<tensor<1x16xf32>>
module {
func.func @test_tosa_sigmoid_to_secret_arith(%vec : !secret.secret<tensor<1x16xf32>>) -> !secret.secret<tensor<1x16xf32>> {
  %out = secret.generic ins (%vec : !secret.secret<tensor<1x16xf32>>) {
  ^bb0(%converted_vec: tensor<1x16xf32>):
    %0 = tosa.sigmoid %converted_vec : (tensor<1x16xf32>) -> tensor<1x16xf32>
    secret.yield %0 : tensor<1x16xf32>
  } -> !secret.secret<tensor<1x16xf32>>
  return %out : !secret.secret<tensor<1x16xf32>>
}
}
