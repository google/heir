// RUN: heir-opt --operation-balancer %s | FileCheck %s

// CHECK:     func.func @simple_balance_secret_add_tree(%[[ARG0:.*]]: !secret.secret<i16>, %[[ARG1:.*]]: !secret.secret<i16>, %[[ARG2:.*]]: !secret.secret<i16>, %[[ARG3:.*]]: !secret.secret<i16>)
// CHECK:     %[[RET:.*]] = secret.generic ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] : !secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>)
// CHECK:     ^bb0(%[[CONVERTED_ARG0:.*]]: i16, %[[CONVERTED_ARG1:.*]]: i16, %[[CONVERTED_ARG2:.*]]: i16, %[[CONVERTED_ARG3:.*]]: i16):
// CHECK-DAG: %[[ADD_ONE:.*]] = arith.addi %[[CONVERTED_ARG0]], %[[CONVERTED_ARG1]]
// CHECK-DAG: %[[ADD_TWO:.*]] = arith.addi %[[CONVERTED_ARG2]], %[[CONVERTED_ARG3]]
// CHECK:     %[[ADD_THREE:.*]] = arith.addi %[[ADD_ONE]], %[[ADD_TWO]]
// CHECK:     secret.yield %[[ADD_THREE]]
// CHECK:     return %[[RET]]
module {
func.func @simple_balance_secret_add_tree(%arg0: !secret.secret<i16>, %arg1 : !secret.secret<i16>,
    %arg2 : !secret.secret<i16>, %arg3 : !secret.secret<i16>) -> !secret.secret<i16> {
  %0 = secret.generic ins (%arg0, %arg1, %arg2, %arg3 : !secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>, !secret.secret<i16>) {
  ^bb0(%converted_arg0: i16, %converted_arg1: i16, %converted_arg2: i16, %converted_arg3: i16):
    %1 = arith.addi %converted_arg0, %converted_arg1 :i16
    %2 = arith.addi %1, %converted_arg2 : i16
    %3 = arith.addi %2, %converted_arg3 : i16
    secret.yield %3 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
}
