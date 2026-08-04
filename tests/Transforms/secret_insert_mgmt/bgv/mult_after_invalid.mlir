// RUN: not heir-opt %s --secret-insert-mgmt-bgv="after-mul=true level-budget=1" 2>&1 | FileCheck %s

module {
  func.func @mult(%arg0: !secret.secret<i16>) -> !secret.secret<i16> {
    %0 = secret.generic(%arg0 : !secret.secret<i16>) {
    ^body(%input0: i16):
      // CHECK: error: value has invalid level
      %1 = arith.muli %input0, %input0 : i16
      %2 = arith.addi %1, %1 : i16
      %3 = arith.muli %2, %2 : i16
      secret.yield %3 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
  }
}
