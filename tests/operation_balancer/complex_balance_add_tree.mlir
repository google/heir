// This test checks that the OperationBalancer can balance a tree with
// intermediate values and disconnected graphs.

// RUN: heir-opt --operation-balancer %s | FileCheck %s

// CHECK:     func.func @complex_balance_add_tree(%[[ARG0:.*]]: i16, %[[ARG1:.*]]: i16, %[[ARG2:.*]]: i16, %[[ARG3:.*]]: i16, %[[ARG4:.*]]: i16, %[[ARG5:.*]]: i16, %[[ARG6:.*]]: i16)
// CHECK-DAG: %[[ADD_ONE:.*]] = arith.addi %[[ARG0]], %[[ARG1]]
// CHECK-DAG: %[[ADD_TWO:.*]] = arith.addi %[[ARG2]], %[[ARG3]]
// CHECK:     %[[ADD_THREE:.*]] = arith.addi %[[ADD_ONE]], %[[ADD_TWO]]

// CHECK-DAG: %[[ADD_FOUR:.*]] = arith.addi %[[ADD_THREE]], %[[ARG4]]
// CHECK-DAG: %[[ADD_FIVE:.*]] = arith.addi %[[ARG5]], %[[ARG6]]
// CHECK:     %[[ADD_SIX:.*]] = arith.addi %[[ADD_FOUR]], %[[ADD_FIVE]]

// CHECK-DAG: %[[ADD_SEVEN:.*]] = arith.addi %[[ARG2]], %[[ARG3]]
// CHECK-DAG: %[[ADD_EIGHT:.*]] = arith.addi %[[ARG4]], %[[ARG5]]
// CHECK:     %[[ADD_NINE:.*]] = arith.addi %[[ADD_SEVEN]], %[[ADD_EIGHT]]
func.func @complex_balance_add_tree(%arg0: i16, %arg1 : i16,
    %arg2 : i16, %arg3 : i16, %arg4 : i16, %arg5 : i16,
    %arg6 : i16) -> (i16, i16, i16) {
  %1 = arith.addi %arg0, %arg1 : i16
  %2 = arith.addi %1, %arg2 : i16
  %3 = arith.addi %2, %arg3 : i16

  %4 = arith.addi %3, %arg4 : i16
  %5 = arith.addi %4, %arg5 : i16
  %6 = arith.addi %5, %arg6 : i16

  %7 = arith.addi %arg4, %arg5 : i16
  %8 = arith.addi %arg3, %7 : i16
  %9 = arith.addi %arg2, %8 : i16
  return %3, %6, %9 : i16, i16, i16
}
