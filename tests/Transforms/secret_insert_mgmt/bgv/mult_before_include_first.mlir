// RUN: heir-opt %s --split-input-file --secret-insert-mgmt-bgv=before-mul-include-first-mul | FileCheck %s

// CHECK: module attributes {scheme.bgv}
module {
  // CHECK-LABEL: func @mult
  func.func @mult(%arg0: !secret.secret<i16>) -> !secret.secret<i16> {
    // CHECK: level = 3
    %0 = secret.generic ins(%arg0 : !secret.secret<i16>) {
    // CHECK: ^body(%[[INPUT0:.*]]: i16):
    ^body(%input0: i16):
      // CHECK: %[[v1:.*]] = mgmt.modreduce %[[INPUT0]]
      // CHECK-NEXT: %[[v2:.*]] = arith.muli %[[v1]], %[[v1]]
      // CHECK-NEXT: %[[v3:.*]] = mgmt.relinearize %[[v2]]
      // CHECK-NEXT: %[[v4:.*]] = arith.addi %[[v3]], %[[v3]]
      // CHECK-NEXT: %[[v5:.*]] = mgmt.modreduce %[[v4]]
      // CHECK-NEXT: %[[v6:.*]] = arith.muli %[[v5]], %[[v5]]
      // CHECK-NEXT: %[[v7:.*]] = mgmt.relinearize %[[v6]]
      // CHECK-NEXT: %[[v8:.*]] = mgmt.modreduce %[[v7]]
      // CHECK-NEXT: secret.yield %[[v8]]
      %1 = arith.muli %input0, %input0 : i16
      %2 = arith.addi %1, %1 : i16
      %3 = arith.muli %2, %2 : i16
      secret.yield %3 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
  }
}

// -----

// MatchCrossLevel

// CHECK: module attributes {scheme.bgv}
module {
  // CHECK-LABEL: func @mul
  func.func @mul(%arg0: !secret.secret<i16>) -> !secret.secret<i16> {
    // CHECK: level = 2
    %0 = secret.generic ins(%arg0 : !secret.secret<i16>) {
    // CHECK: ^body(%[[INPUT0:.*]]: i16):
    ^body(%input0: i16):
      // CHECK: %[[v1:.*]] = mgmt.modreduce %[[INPUT0]]
      // CHECK-NEXT: %[[v2:.*]] = arith.muli %[[v1]], %[[v1]]
      // CHECK-NEXT: %[[v3:.*]] = mgmt.relinearize %[[v2]]
      %1 = arith.muli %input0, %input0 : i16
      // The addition of %1 and %input0 may mismatch scale because they are at different _level_.
      // So MatchCrossLevel is effective here
      // CHECK-NEXT: %[[v4:.*]] = mgmt.adjust_scale %[[INPUT0]]
      // CHECK-NEXT: %[[v5:.*]] = mgmt.modreduce %[[v4]]
      // CHECK-NEXT: %[[v6:.*]] = arith.addi %[[v3]], %[[v5]]
      %2 = arith.addi %1, %input0 : i16
      // CHECK-NEXT: %[[v7:.*]] = mgmt.modreduce %[[v6]]
      // CHECK-NEXT: secret.yield %[[v7]]
      secret.yield %2 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
  }
}
