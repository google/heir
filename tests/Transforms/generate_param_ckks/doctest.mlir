// RUN: heir-opt --generate-param-ckks %s | FileCheck %s

// CHECK: {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673], P = [1152921504606994433], logDefaultScale = 45>
module {
  func.func @add(%arg0: !secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) -> (!secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) {
    %0 = secret.generic(%arg0: !secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 0>}} {
    ^body(%input0: f16):
      %1 = arith.addf %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : f16
      secret.yield %1 : f16
    } -> (!secret.secret<f16> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
    return %0 : !secret.secret<f16>
  }
}
