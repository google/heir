// RUN: heir-opt --validate-scale --verify-diagnostics --split-input-file %s

module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 12, Q = [67239937, 34359754753], P = [34359771137], plaintextModulus = 65537>, scheme.bgv} {
  func.func @bgv_addi_mismatch(%arg0: !secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 1>}, %arg1: !secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 2>}) -> !secret.secret<i16> {
    %0 = secret.generic(%arg0 : !secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 1>}, %arg1 : !secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 2>}) {
    ^body(%input0: i16, %input1: i16):
      // expected-error @+1 {{operands and results must have all the same scale}}
      %1 = arith.addi %input0, %input1 {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 2>} : i16
      secret.yield %1 : i16
    } -> (!secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 2>})
    return %0 : !secret.secret<i16>
  }
}
