// This test checks that the OperationBalancer can balance a tree with
// intermediate values and disconnected graphs.

// RUN: heir-opt --operation-balancer %s | FileCheck %s

// CHECK:     func.func @complex_balance_secret_mult_tree(%[[ARG0:.*]]: !secret.secret<i16>, %[[ARG1:.*]]: !secret.secret<i16>, %[[ARG2:.*]]: !secret.secret<i16>, %[[ARG3:.*]]: !secret.secret<i16>, %[[ARG4:.*]]: !secret.secret<i16>, %[[ARG5:.*]]: !secret.secret<i16>, %[[ARG6:.*]]: !secret.secret<i16>)

// CHECK:     %[[RET:.*]]:3 = secret.generic ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]] : !secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>)
// CHECK:     ^bb0(%[[CONVERTED_ARG0:.*]]: i16, %[[CONVERTED_ARG1:.*]]: i16, %[[CONVERTED_ARG2:.*]]: i16, %[[CONVERTED_ARG3:.*]]: i16, %[[CONVERTED_ARG4:.*]]: i16, %[[CONVERTED_ARG5:.*]]: i16, %[[CONVERTED_ARG6:.*]]: i16):

// CHECK-DAG: %[[MULT_ONE:.*]] = arith.muli %[[CONVERTED_ARG0]], %[[CONVERTED_ARG1]]
// CHECK-DAG: %[[MULT_TWO:.*]] = arith.muli %[[CONVERTED_ARG2]], %[[CONVERTED_ARG3]]
// CHECK:     %[[MULT_THREE:.*]] = arith.muli %[[MULT_ONE]], %[[MULT_TWO]]

// CHECK-DAG: %[[MULT_FOUR:.*]] = arith.muli %[[MULT_THREE]], %[[CONVERTED_ARG4]]
// CHECK-DAG: %[[MULT_FIVE:.*]] = arith.muli %[[CONVERTED_ARG5]], %[[CONVERTED_ARG6]]
// CHECK:     %[[MULT_SIX:.*]] = arith.muli %[[MULT_FOUR]], %[[MULT_FIVE]]

// CHECK-DAG: %[[MULT_SEVEN:.*]] = arith.muli %[[CONVERTED_ARG2]], %[[CONVERTED_ARG3]]
// CHECK-DAG: %[[MULT_EIGHT:.*]] = arith.muli %[[CONVERTED_ARG4]], %[[CONVERTED_ARG5]]
// CHECK:     %[[MULT_NINE:.*]] = arith.muli %[[MULT_SEVEN]], %[[MULT_EIGHT]]

// CHECK:     secret.yield %[[MULT_THREE]], %[[MULT_SIX]], %[[MULT_NINE]]
// CHECK:     return %[[RET]]#0, %[[RET]]#1, %[[RET]]#2

module {
func.func @complex_balance_secret_mult_tree(%arg0: !secret.secret<i16>, %arg1 : !secret.secret<i16>,
    %arg2 : !secret.secret<i16>, %arg3 : !secret.secret<i16>, %arg4 : !secret.secret<i16>, %arg5 : !secret.secret<i16>,
    %arg6 : !secret.secret<i16>) -> (!secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>) {
  %out0, %out1, %out2 = secret.generic ins (%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6 :
      !secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>,
      !secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>,
      !secret.secret<i16>) {
  ^bb0(%converted_arg0: i16, %converted_arg1: i16, %converted_arg2: i16,
      %converted_arg3: i16, %converted_arg4: i16, %converted_arg5: i16,
      %converted_arg6: i16):
    %1 = arith.muli %converted_arg0, %converted_arg1 : i16
    %2 = arith.muli %1, %converted_arg2 : i16
    %3 = arith.muli %2, %converted_arg3 : i16

    %4 = arith.muli %3, %converted_arg4 : i16
    %5 = arith.muli %4, %converted_arg5 : i16
    %6 = arith.muli %5, %converted_arg6 : i16

    %7 = arith.muli %converted_arg4, %converted_arg5 : i16
    %8 = arith.muli %converted_arg3, %7 : i16
    %9 = arith.muli %converted_arg2, %8 : i16
    secret.yield %3, %6, %9 : i16, i16, i16
  } -> (!secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>)
  return %out0, %out1, %out2 : !secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>
}
}
