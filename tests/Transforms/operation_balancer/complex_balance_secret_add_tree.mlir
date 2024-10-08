// This test checks that the OperationBalancer can balance a tree with
// intermediate values and disconnected graphs.

// RUN: heir-opt --operation-balancer %s | FileCheck %s

// CHECK:     func.func @complex_balance_secret_add_tree(%[[ARG0:.*]]: !secret.secret<i16>, %[[ARG1:.*]]: !secret.secret<i16>, %[[ARG2:.*]]: !secret.secret<i16>, %[[ARG3:.*]]: !secret.secret<i16>, %[[ARG4:.*]]: !secret.secret<i16>, %[[ARG5:.*]]: !secret.secret<i16>, %[[ARG6:.*]]: !secret.secret<i16>)

// CHECK:     %[[RET:.*]]:3 = secret.generic ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]] : !secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>)
// CHECK:     ^bb0(%[[CONVERTED_ARG0:.*]]: i16, %[[CONVERTED_ARG1:.*]]: i16, %[[CONVERTED_ARG2:.*]]: i16, %[[CONVERTED_ARG3:.*]]: i16, %[[CONVERTED_ARG4:.*]]: i16, %[[CONVERTED_ARG5:.*]]: i16, %[[CONVERTED_ARG6:.*]]: i16):

// CHECK-DAG: %[[ADD_ONE:.*]] = arith.addi %[[CONVERTED_ARG0]], %[[CONVERTED_ARG1]]
// CHECK-DAG: %[[ADD_TWO:.*]] = arith.addi %[[CONVERTED_ARG2]], %[[CONVERTED_ARG3]]
// CHECK:     %[[ADD_THREE:.*]] = arith.addi %[[ADD_ONE]], %[[ADD_TWO]]

// CHECK-DAG: %[[ADD_FOUR:.*]] = arith.addi %[[ADD_THREE]], %[[CONVERTED_ARG4]]
// CHECK-DAG: %[[ADD_FIVE:.*]] = arith.addi %[[CONVERTED_ARG5]], %[[CONVERTED_ARG6]]
// CHECK:     %[[ADD_SIX:.*]] = arith.addi %[[ADD_FOUR]], %[[ADD_FIVE]]

// CHECK-DAG: %[[ADD_SEVEN:.*]] = arith.addi %[[CONVERTED_ARG2]], %[[CONVERTED_ARG3]]
// CHECK-DAG: %[[ADD_EIGHT:.*]] = arith.addi %[[CONVERTED_ARG4]], %[[CONVERTED_ARG5]]
// CHECK:     %[[ADD_NINE:.*]] = arith.addi %[[ADD_SEVEN]], %[[ADD_EIGHT]]

// CHECK:     secret.yield %[[ADD_THREE]], %[[ADD_SIX]], %[[ADD_NINE]]
// CHECK:     return %[[RET]]#0, %[[RET]]#1, %[[RET]]#2

module {
func.func @complex_balance_secret_add_tree(%arg0: !secret.secret<i16>, %arg1 : !secret.secret<i16>,
    %arg2 : !secret.secret<i16>, %arg3 : !secret.secret<i16>, %arg4 : !secret.secret<i16>, %arg5 : !secret.secret<i16>,
    %arg6 : !secret.secret<i16>) -> (!secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>) {
  %out0, %out1, %out2 = secret.generic ins (%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6 :
      !secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>,
      !secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>,
      !secret.secret<i16>) {
  ^bb0(%converted_arg0: i16, %converted_arg1: i16, %converted_arg2: i16,
      %converted_arg3: i16, %converted_arg4: i16, %converted_arg5: i16,
      %converted_arg6: i16):
    %1 = arith.addi %converted_arg0, %converted_arg1 : i16
    %2 = arith.addi %1, %converted_arg2 : i16
    %3 = arith.addi %2, %converted_arg3 : i16

    %4 = arith.addi %3, %converted_arg4 : i16
    %5 = arith.addi %4, %converted_arg5 : i16
    %6 = arith.addi %5, %converted_arg6 : i16

    %7 = arith.addi %converted_arg4, %converted_arg5 : i16
    %8 = arith.addi %converted_arg3, %7 : i16
    %9 = arith.addi %converted_arg2, %8 : i16
    secret.yield %3, %6, %9 : i16, i16, i16
  } -> (!secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>)
  return %out0, %out1, %out2 : !secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>
}
}
