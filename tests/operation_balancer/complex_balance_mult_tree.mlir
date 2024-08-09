// This test checks that the OperationBalancer can balance a tree with
// intermediate values and disconnected graphs.

// RUN: heir-opt --operation-balancer %s | FileCheck %s

// CHECK:     func.func @complex_balance_mult_tree(%[[ARG0:.*]]: i16, %[[ARG1:.*]]: i16, %[[ARG2:.*]]: i16, %[[ARG3:.*]]: i16, %[[ARG4:.*]]: i16, %[[ARG5:.*]]: i16, %[[ARG6:.*]]: i16)
// CHECK-DAG: %[[MULT_ONE:.*]] = arith.muli %[[ARG0]], %[[ARG1]]
// CHECK-DAG: %[[MULT_TWO:.*]] = arith.muli %[[ARG2]], %[[ARG3]]
// CHECK:     %[[MULT_THREE:.*]] = arith.muli %[[MULT_ONE]], %[[MULT_TWO]]

// CHECK-DAG: %[[MULT_FOUR:.*]] = arith.muli %[[MULT_THREE]], %[[ARG4]]
// CHECK-DAG: %[[MULT_FIVE:.*]] = arith.muli %[[ARG5]], %[[ARG6]]
// CHECK:     %[[MULT_SIX:.*]] = arith.muli %[[MULT_FOUR]], %[[MULT_FIVE]]

// CHECK-DAG: %[[MULT_SEVEN:.*]] = arith.muli %[[ARG2]], %[[ARG3]]
// CHECK-DAG: %[[MULT_EIGHT:.*]] = arith.muli %[[ARG4]], %[[ARG5]]
// CHECK:     %[[MULT_NINE:.*]] = arith.muli %[[MULT_SEVEN]], %[[MULT_EIGHT]]
func.func @complex_balance_mult_tree(%arg0: i16, %arg1 : i16,
    %arg2 : i16, %arg3 : i16, %arg4 : i16, %arg5 : i16,
    %arg6 : i16) -> (i16, i16, i16) {
  %1 = arith.muli %arg0, %arg1 : i16
  %2 = arith.muli %1, %arg2 : i16
  %3 = arith.muli %2, %arg3 : i16

  %4 = arith.muli %3, %arg4 : i16
  %5 = arith.muli %4, %arg5 : i16
  %6 = arith.muli %5, %arg6 : i16

  %7 = arith.muli %arg4, %arg5 : i16
  %8 = arith.muli %arg3, %7 : i16
  %9 = arith.muli %arg2, %8 : i16
  return %3, %6, %9 : i16, i16, i16
}
