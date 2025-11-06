// RUN: heir-opt --generate-param-bgv %s | FileCheck %s

// CHECK: module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 12, Q = [4294991873], P = [4295049217], plaintextModulus = 65537>}
module {
  func.func @add(%arg0: !secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>}) -> (!secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>}) {
    %0 = secret.generic(%arg0: !secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>}) attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>}} {
    ^body(%input0: i16):
      %1 = arith.addi %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>} : i16
      secret.yield %1 : i16
    } -> (!secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>})
    return %0 : !secret.secret<i16>
  }
}
