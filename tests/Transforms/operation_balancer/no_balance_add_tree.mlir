// This test checks that the OperationBalancer does not balance a tree with operations
// not encapsulated in a secret.generic.

// RUN: heir-opt --operation-balancer %s | FileCheck %s

// CHECK:     func.func @no_balance_add_tree(%[[ARG0:.*]]: i16, %[[ARG1:.*]]: i16, %[[ARG2:.*]]: i16, %[[ARG3:.*]]: i16)
// CHECK-DAG: %[[ADD_ONE:.*]] = arith.addi %[[ARG0]], %[[ARG1]]
// CHECK-DAG: %[[ADD_TWO:.*]] = arith.addi %[[ADD_ONE]], %[[ARG2]]
// CHECK:     %[[ADD_THREE:.*]] = arith.addi %[[ADD_TWO]], %[[ARG3]]
// CHECK:     return %[[ADD_THREE]]
func.func @no_balance_add_tree(%arg0: i16, %arg1 : i16, %arg2 : i16, %arg3 : i16) -> (i16) {
  %1 = arith.addi %arg0, %arg1 :i16
  %2 = arith.addi %1, %arg2 : i16
  %3 = arith.addi %2, %arg3 : i16
  return %3 : i16
}
