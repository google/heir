// RUN: heir-opt %s --split-input-file --secret-insert-mgmt-bgv --populate-scale-bgv | FileCheck %s

module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 13, Q = [134250497, 17179967489, 35184372121601], P = [35184372203521, 35184372744193], plaintextModulus = 65537>, scheme.bgv} {
  // CHECK-LABEL: func @mult
  func.func @mult(%arg0: !secret.secret<i16>) -> !secret.secret<i16> {
    // CHECK: __argattrs
    // CHECK-SAME: level = 2
    // CHECK-SAME: scale = 1
    // CHECK-SAME: __resattrs
    // CHECK-SAME: level = 0
    // CHECK-SAME: scale = 11445
    %0 = secret.generic ins(%arg0 : !secret.secret<i16>) {
    ^body(%input0: i16):
      %1 = arith.muli %input0, %input0 : i16
      %2 = arith.addi %1, %1 : i16
      %3 = arith.muli %2, %2 : i16
      secret.yield %3 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
  }
}

// -----

// MatchCrossMulDepth and their actual scale is the same

module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 13, Q = [67239937, 8796093202433], P = [8796093349889], plaintextModulus = 65537>, scheme.bgv} {
  // CHECK-LABEL: func @mul
  func.func @mul(%arg0: !secret.secret<i16>) -> !secret.secret<i16> {
    %0 = secret.generic ins(%arg0 : !secret.secret<i16>) {
    // CHECK: ^body(%[[INPUT0:.*]]: i16):
    ^body(%input0: i16):
      // CHECK: %[[v1:.*]] = arith.muli %[[INPUT0]], %[[INPUT0]]
      // CHECK-NEXT: %[[v2:.*]] = mgmt.relinearize %[[v1]]
      %1 = arith.muli %input0, %input0 : i16
      // CHECK-NEXT: %[[v3:.*]] = arith.addi %[[v2]], %[[INPUT0]]
      %2 = arith.addi %1, %input0 : i16
      // CHECK-NEXT: %[[v4:.*]] = mgmt.modreduce %[[v3]]
      // CHECK-NEXT: secret.yield %[[v4]]
      secret.yield %2 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
  }
}
