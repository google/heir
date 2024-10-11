// This test checks that the OperationBalancer does not balance a tree with operations
// not encapsulated in a secret.generic.

// RUN: heir-opt --operation-balancer %s | FileCheck %s

// CHECK:     func.func @no_balance_mult_tree(%[[ARG0:.*]]: i16, %[[ARG1:.*]]: i16, %[[ARG2:.*]]: i16, %[[ARG3:.*]]: i16)
// CHECK-DAG: %[[MULT_ONE:.*]] = arith.muli %[[ARG0]], %[[ARG1]]
// CHECK-DAG: %[[MULT_TWO:.*]] = arith.muli %[[MULT_ONE]], %[[ARG2]]
// CHECK:     %[[MULT_THREE:.*]] = arith.muli %[[MULT_TWO]], %[[ARG3]]
// CHECK:     return %[[MULT_THREE]]
func.func @no_balance_mult_tree(%arg0: i16, %arg1 : i16, %arg2 : i16, %arg3 : i16) -> (i16) {
  %1 = arith.muli %arg0, %arg1 :i16
  %2 = arith.muli %1, %arg2 : i16
  %3 = arith.muli %2, %arg3 : i16
  return %3 : i16
}
