// RUN: heir-opt --secret-insert-mgmt-bfv %s | FileCheck %s

// CHECK: func.func @func
// CHECK:      %[[GENERIC:.*]] = secret.generic(%{{.*}}, %{{.*}}) {
// CHECK:      ^body(%[[ARG0:.*]]: i16, %[[ARG1:.*]]: i16):
// CHECK-NEXT:   %[[MUL:.*]] = arith.muli %[[ARG0]], %[[ARG1]] {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : i16
// CHECK-NEXT:   %[[RELIN:.*]] = mgmt.relinearize %[[MUL]] {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i16
// CHECK-NEXT:   %[[ADD:.*]] = arith.addi %[[RELIN]], %[[ARG1]] {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i16
// CHECK-NEXT:   secret.yield %[[ADD]] : i16
// CHECK:      return
func.func @func(%arg0: !secret.secret<i16>, %arg1: !secret.secret<i16>) -> !secret.secret<i16> {
  %0 = secret.generic(%arg0: !secret.secret<i16>, %arg1: !secret.secret<i16>) {
  ^bb0(%arg2: i16, %arg3: i16):
    %1 = arith.muli %arg2, %arg3 : i16
    %2 = arith.addi %1, %arg3 : i16
    secret.yield %2 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
