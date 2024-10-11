// RUN: heir-opt --operation-balancer %s | FileCheck %s

// CHECK:     func.func @simple_balance_secret_add_float_tree(%[[ARG0:.*]]: !secret.secret<f32>, %[[ARG1:.*]]: !secret.secret<f32>, %[[ARG2:.*]]: !secret.secret<f32>, %[[ARG3:.*]]: !secret.secret<f32>)
// CHECK:     %[[RET:.*]] = secret.generic ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] : !secret.secret<f32>, !secret.secret<f32>, !secret.secret<f32>, !secret.secret<f32>)
// CHECK:     ^bb0(%[[CONVERTED_ARG0:.*]]: f32, %[[CONVERTED_ARG1:.*]]: f32, %[[CONVERTED_ARG2:.*]]: f32, %[[CONVERTED_ARG3:.*]]: f32):
// CHECK-DAG: %[[ADD_ONE:.*]] = arith.addf %[[CONVERTED_ARG0]], %[[CONVERTED_ARG1]]
// CHECK-DAG: %[[ADD_TWO:.*]] = arith.addf %[[CONVERTED_ARG2]], %[[CONVERTED_ARG3]]
// CHECK:     %[[ADD_THREE:.*]] = arith.addf %[[ADD_ONE]], %[[ADD_TWO]]
// CHECK:     secret.yield %[[ADD_THREE]]
// CHECK:     return %[[RET]]
module {
func.func @simple_balance_secret_add_float_tree(%arg0: !secret.secret<f32>, %arg1 : !secret.secret<f32>,
    %arg2 : !secret.secret<f32>, %arg3 : !secret.secret<f32>) -> !secret.secret<f32> {
  %0 = secret.generic ins (%arg0, %arg1, %arg2, %arg3 : !secret.secret<f32>, !secret.secret<f32>, !secret.secret<f32>, !secret.secret<f32>) {
  ^bb0(%converted_arg0: f32, %converted_arg1: f32, %converted_arg2: f32, %converted_arg3: f32):
    %1 = arith.addf %converted_arg0, %converted_arg1 :f32
    %2 = arith.addf %1, %converted_arg2 : f32
    %3 = arith.addf %2, %converted_arg3 : f32
    secret.yield %3 : f32
  } -> !secret.secret<f32>
  return %0 : !secret.secret<f32>
}
}
