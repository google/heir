// RUN: heir-opt %s "--annotate-mgmt" | FileCheck %s

module attributes {backend.openfhe, scheme.bfv} {
  // CHECK: @add
  // CHECK-SAME: (%[[arg0:[^:]*]]: !secret.secret<i16> {mgmt.mgmt =
  // CHECK-SAME: %[[arg1:[^:]*]]: !secret.secret<i16> {mgmt.mgmt =
  // CHECK-SAME: -> (!secret.secret<i16> {mgmt.mgmt =
  func.func @add(%arg0: !secret.secret<i16>, %arg1: !secret.secret<i16>) -> !secret.secret<i16> {
    %0 = secret.generic(%arg0: !secret.secret<i16>) {
    ^body(%input0: i16):
      %1 = arith.addi %input0, %input0 : i16
      secret.yield %1 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
  }
}
