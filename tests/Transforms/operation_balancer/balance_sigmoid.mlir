// RUN: heir-opt --operation-balancer %s | FileCheck %s

// CHECK:     func.func @three_degree_sigmoid(%[[ARG0:.*]]: !secret.secret<tensor<1x16xf32>>)

// CHECK-DAG: %[[COEFF_3:.*]] = arith.constant dense<-4.{{0*}}e-03> : tensor<1x16xf32>
// CHECK-DAG: %[[COEFF_1:.*]] = arith.constant dense<1.97{{0*}}e-01> : tensor<1x16xf32>
// CHECK-DAG: %[[COEFF_0:.*]] = arith.constant dense<5.{{0*}}e-01> : tensor<1x16xf32>

// CHECK:     %[[RET:.*]] = secret.generic ins(%[[ARG0]] : !secret.secret<tensor<1x16xf32>>)
// CHECK:     ^bb0(%[[CONVERTED_ARG:.*]]: tensor<1x16xf32>):
// CHECK:       %[[COEFF_MUL_DEGREE_1:.*]] = arith.mulf %[[CONVERTED_ARG]], %[[COEFF_1]]

// CHECK:       %[[DEGREE_2:.*]] = arith.mulf %[[CONVERTED_ARG]], %[[CONVERTED_ARG]]
// CHECK:       %[[COEFF_3_MUL_ARG:.*]] = arith.mulf %[[CONVERTED_ARG]], %[[COEFF_3]]
// CHECK:       %[[COEFF_MUL_DEGREE_3:.*]] = arith.mulf %[[DEGREE_2]], %[[COEFF_3_MUL_ARG]]

// CHECK:       %[[SUM_1:.*]] = arith.addf %[[COEFF_MUL_DEGREE_1]], %[[COEFF_0]]
// CHECK:       %[[TOTAL_SUM:.*]] = arith.addf %[[SUM_1]], %[[COEFF_MUL_DEGREE_3]]
// CHECK:       secret.yield %[[TOTAL_SUM]] : tensor<1x16xf32>
// CHECK:     return %[[RET]]

module {
  func.func @three_degree_sigmoid(%arg0: !secret.secret<tensor<1x16xf32>>) -> !secret.secret<tensor<1x16xf32>> {
    %cst = arith.constant dense<-4.000000e-03> : tensor<1x16xf32>
    %cst_0 = arith.constant dense<1.970000e-01> : tensor<1x16xf32>
    %cst_1 = arith.constant dense<5.000000e-01> : tensor<1x16xf32>
    %1 = secret.generic ins(%arg0 : !secret.secret<tensor<1x16xf32>>) {
    ^bb0(%arg1: tensor<1x16xf32>):
      %5 = arith.mulf %arg1, %cst_0 : tensor<1x16xf32>
      %6 = arith.mulf %arg1, %arg1 : tensor<1x16xf32>
      %7 = arith.mulf %6, %arg1 : tensor<1x16xf32>
      %8 = arith.mulf %7, %cst : tensor<1x16xf32>
      %9 = arith.addf %5, %cst_1 : tensor<1x16xf32>
      %10 = arith.addf %9, %8 : tensor<1x16xf32>
      secret.yield %10 : tensor<1x16xf32>
    } -> !secret.secret<tensor<1x16xf32>>
    return %1 : !secret.secret<tensor<1x16xf32>>
  }
}
