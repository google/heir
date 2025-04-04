// RUN: heir-opt %s --split-input-file --secret-insert-mgmt-bgv | FileCheck %s

// CHECK: module attributes {scheme.bgv}
module {
  // CHECK-LABEL: func @mult
  func.func @mult(%arg0: !secret.secret<i16>) -> !secret.secret<i16> {
    // CHECK: level = 2
    %0 = secret.generic ins(%arg0 : !secret.secret<i16>) {
    // CHECK: ^body(%[[INPUT0:.*]]: i16):
    ^body(%input0: i16):
      // CHECK: %[[v1:.*]] = arith.muli %[[INPUT0]], %[[INPUT0]]
      // CHECK-NEXT: %[[v2:.*]] = mgmt.relinearize %[[v1]]
      // CHECK-NEXT: %[[v3:.*]] = arith.addi %[[v2]], %[[v2]]
      // CHECK-NEXT: %[[v4:.*]] = mgmt.modreduce %[[v3]]
      // CHECK-NEXT: %[[v5:.*]] = arith.muli %[[v4]], %[[v4]]
      // CHECK-NEXT: %[[v6:.*]] = mgmt.relinearize %[[v5]]
      // CHECK-NEXT: %[[v7:.*]] = mgmt.modreduce %[[v6]]
      // CHECK-NEXT: secret.yield %[[v7]]
      %1 = arith.muli %input0, %input0 : i16
      %2 = arith.addi %1, %1 : i16
      %3 = arith.muli %2, %2 : i16
      secret.yield %3 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
  }
}

// -----

// MatchCrossMulDepth

// CHECK: module attributes {scheme.bgv}
module {
  // CHECK-LABEL: func @mul
  func.func @mul(%arg0: !secret.secret<i16>) -> !secret.secret<i16> {
    // CHECK: level = 1
    %0 = secret.generic ins(%arg0 : !secret.secret<i16>) {
    // CHECK: ^body(%[[INPUT0:.*]]: i16):
    ^body(%input0: i16):
      // CHECK: %[[v1:.*]] = arith.muli %[[INPUT0]], %[[INPUT0]]
      // CHECK-NEXT: %[[v2:.*]] = mgmt.relinearize %[[v1]]
      %1 = arith.muli %input0, %input0 : i16
      // The addition of %1 and %input0 may mismatch scale though at the same _level_.
      // So MatchCrossMulDepth is effective here
      // CHECK-NEXT: %[[v3:.*]] = mgmt.adjust_scale %[[INPUT0]]
      // CHECK-NEXT: %[[v4:.*]] = arith.addi %[[v2]], %[[v3]]
      %2 = arith.addi %1, %input0 : i16
      // CHECK-NEXT: %[[v5:.*]] = mgmt.modreduce %[[v4]]
      // CHECK-NEXT: secret.yield %[[v5]]
      secret.yield %2 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
  }
}
