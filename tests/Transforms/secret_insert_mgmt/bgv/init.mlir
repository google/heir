// RUN: heir-opt %s --secret-insert-mgmt-bgv | FileCheck %s

// UseInitOpForPlaintextOperand

// result
// module attributes {scheme.bgv} {
//   func.func @init(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
//     %c4_i32 = arith.constant 4 : i32
//     %c3_i32 = arith.constant 3 : i32
//     %c2_i32 = arith.constant 2 : i32
//     %0 = mgmt.init %c2_i32 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : i32
//     %1 = mgmt.init %c4_i32 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : i32
//     %2 = mgmt.init %c3_i32 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i32
//     %3 = secret.generic ins(%arg0 : !secret.secret<i32>) attrs = {__argattrs = [{mgmt.mgmt = #mgmt.mgmt<level = 2>}], __resattrs = [{mgmt.mgmt = #mgmt.mgmt<level = 0>}]} {
//     ^body(%input0: i32):
//       %4 = arith.muli %input0, %0 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : i32
//       %5 = arith.addi %4, %1 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : i32
//       %6 = mgmt.modreduce %5 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i32
//       %7 = arith.muli %6, %2 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i32
//       %8 = mgmt.modreduce %7 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : i32
//       secret.yield %8 : i32
//     } -> !secret.secret<i32>
//     return %3 : !secret.secret<i32>
//   }
// }

module {
  // CHECK: @init
  func.func @init(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
    // CHECK: %c4_i32 = arith.constant 4 : i32
    // CHECK: %c3_i32 = arith.constant 3 : i32
    // CHECK: %c2_i32 = arith.constant 2 : i32
    // CHECK: %[[v0:.*]] = mgmt.init %c2_i32 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : i32
    // CHECK: %[[v1:.*]] = mgmt.init %c4_i32 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : i32
    // CHECK: %[[v2:.*]] = mgmt.init %c3_i32 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i32
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = secret.generic ins(%arg0 : !secret.secret<i32>) {
    // CHECK: ^body(%[[INPUT0:.*]]: i32):
    ^body(%input0: i32):
      // CHECK: %[[v3:.*]] = arith.muli %[[INPUT0]], %[[v0]] {mgmt.mgmt = #mgmt.mgmt<level = 2>} : i32
      %1 = arith.muli %input0, %c2_i32 : i32
      // CHECK: %[[v4:.*]] = arith.addi %[[v3]], %[[v1]] {mgmt.mgmt = #mgmt.mgmt<level = 2>} : i32
      %2 = arith.addi %1, %c4_i32 : i32
      // CHECK: %[[v5:.*]] = mgmt.modreduce %[[v4]] {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i32
      // CHECK: %[[v6:.*]] = arith.muli %[[v5]], %[[v2]] {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i32
      %3 = arith.muli %2, %c3_i32 : i32
      // CHECK: %[[v7:.*]] = mgmt.modreduce %[[v6]] {mgmt.mgmt = #mgmt.mgmt<level = 0>} : i32
      // CHECK: secret.yield %[[v7]] : i32
      secret.yield %3 : i32
    } -> !secret.secret<i32>
    return %0 : !secret.secret<i32>
  }
}
