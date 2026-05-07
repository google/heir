// f(s1, s2, p1) = (s1 * c0 * s2 * p1 * c1) + (s1 + c0 + s2 + p1 + c1)

// RUN: heir-opt --operation-balancer %s | FileCheck %s

// CHECK:       func.func @secrecy_aware_balance(%[[ARG0:.*]]: !secret.secret<tensor<1x16xf32>>, %[[ARG1:.*]]: !secret.secret<tensor<1x16xf32>>, %[[ARG2:.*]]: tensor<1x16xf32>)

// CHECK-DAG: %[[CONST_0:.*]] = arith.constant dense<5.000000e-01> : tensor<1x16xf32>
// CHECK-DAG: %[[CONST_1:.*]] = arith.constant dense<5.000000e-02> : tensor<1x16xf32>

// CHECK:       %[[RET:.*]] = secret.generic(%{{[^:]*}}: !secret.secret<tensor<1x16xf32>>, %{{[^:]*}}: !secret.secret<tensor<1x16xf32>>, %{{[^:]*}}: tensor<1x16xf32>)
// CHECK:       ^body(%[[CONVERTED_ARG0:.*]]: tensor<1x16xf32>, %[[CONVERTED_ARG1:.*]]: tensor<1x16xf32>, %[[CONVERTED_ARG2:.*]]: tensor<1x16xf32>)

// CHECK:           %[[MUL_ONE:.*]] = arith.mulf %[[CONST_0]], %[[CONVERTED_ARG2]] : tensor<1x16xf32>
// CHECK:           %[[MUL_TWO:.*]] = arith.mulf %[[CONST_1]], %[[CONVERTED_ARG0]] : tensor<1x16xf32>
// CHECK:           %[[MUL_THREE:.*]] = arith.mulf %[[MUL_TWO]], %[[CONVERTED_ARG1]] : tensor<1x16xf32>
// CHECK:           %[[MUL_FOUR:.*]] = arith.mulf %[[MUL_ONE]], %[[MUL_THREE]] : tensor<1x16xf32>
// CHECK:           %[[ADD_ONE:.*]] = arith.addf %[[CONST_0]], %[[CONVERTED_ARG2]] : tensor<1x16xf32>
// CHECK:           %[[ADD_TWO:.*]] = arith.addf %[[ADD_ONE]], %[[CONST_1]] : tensor<1x16xf32>
// CHECK:           %[[ADD_THREE:.*]] = arith.addf %[[MUL_FOUR]], %[[CONVERTED_ARG0]] : tensor<1x16xf32>
// CHECK:           %[[ADD_FOUR:.*]] = arith.addf %[[ADD_THREE]], %[[CONVERTED_ARG1]] : tensor<1x16xf32>
// CHECK:           %[[ADD_FIVE:.*]] = arith.addf %[[ADD_TWO]], %[[ADD_FOUR]] : tensor<1x16xf32>
// CHECK:           secret.yield %[[ADD_FIVE]] : tensor<1x16xf32>

// CHECK:           return %[[RET]]

module {
  func.func @secrecy_aware_balance(%arg0: !secret.secret<tensor<1x16xf32>>, %arg1: !secret.secret<tensor<1x16xf32>>, %arg2: tensor<1x16xf32>) -> !secret.secret<tensor<1x16xf32>> {
    %cst_0 = arith.constant dense<5.000000e-01> : tensor<1x16xf32>
    %cst_1 = arith.constant dense<5.000000e-02> : tensor<1x16xf32>
    %1 = secret.generic(%arg0 : !secret.secret<tensor<1x16xf32>>, %arg1: !secret.secret<tensor<1x16xf32>>, %arg2: tensor<1x16xf32>) {
    ^bb0(%arg3: tensor<1x16xf32>, %arg4: tensor<1x16xf32>, %arg5: tensor<1x16xf32>):
      %2 = arith.mulf %arg3, %cst_0 : tensor<1x16xf32>
      %3 = arith.mulf %2, %arg4 : tensor<1x16xf32>
      %4 = arith.mulf %3, %arg5 : tensor<1x16xf32>
      %5 = arith.mulf %4, %cst_1 : tensor<1x16xf32>
      %6 = arith.addf %arg3, %cst_0 : tensor<1x16xf32>
      %7 = arith.addf %6, %arg4 : tensor<1x16xf32>
      %8 = arith.addf %7, %arg5 : tensor<1x16xf32>
      %9 = arith.addf %8, %cst_1 : tensor<1x16xf32>
      %10 = arith.addf %5, %9 : tensor<1x16xf32>
      secret.yield %10 : tensor<1x16xf32>
    } -> !secret.secret<tensor<1x16xf32>>
    return %1 : !secret.secret<tensor<1x16xf32>>
  }
}
