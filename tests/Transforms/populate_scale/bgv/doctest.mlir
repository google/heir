// RUN: heir-opt %s --populate-scale-bgv | FileCheck %s

module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 13, Q = [67239937, 8796093202433], P = [8796093349889], plaintextModulus = 65537>, scheme.bgv} {
  func.func @mul(%arg0: !secret.secret<i16>) -> !secret.secret<i16> {
    %0 = secret.generic(%arg0 : !secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 0>}) {
    ^body(%input0: i16):
      %1 = arith.muli %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3, scale = 0>} : i16
      // CHECK: mgmt.relinearize
      // CHECK-SAME: scale = 1
      %2 = mgmt.relinearize %1 {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 0>} : i16
      // CHECK: mgmt.modreduce
      // CHECK-SAME: scale = 42541
      %3 = mgmt.modreduce %2 {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>} : i16
      secret.yield %3 : i16
    } -> (!secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>})
    return %0 : !secret.secret<i16>
  }
}
