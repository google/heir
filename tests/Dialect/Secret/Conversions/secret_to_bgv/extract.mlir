// RUN: heir-opt --secret-to-bgv %s | FileCheck %s

// CHECK-LABEL: @hamming
// CHECK: bgv.add
// CHECK-NEXT: bgv.extract
// CHECK-NEXT: bgv.modulus_switch
// CHECK-NEXT: return

module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 14, Q = [67239937, 17179967489, 17180262401, 17180295169, 17180393473, 70368744210433], P = [70368744570881, 70368744701953], plaintextModulus = 65537>} {
  func.func @hamming(%arg0: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 1>}, %arg1: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 1>}) -> (!secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) {
    %c0 = arith.constant 0 : index
    %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<1024xi16>>, !secret.secret<tensor<1024xi16>>) attrs = {__resattrs = [{mgmt.mgmt = #mgmt.mgmt<level = 1>}]} {
    ^body(%input0: tensor<1024xi16>, %input1: tensor<1024xi16>):
      %3 = arith.addi %input0, %input1 : tensor<1024xi16>
      secret.yield %3 : tensor<1024xi16>
    } -> !secret.secret<tensor<1024xi16>>
    %1 = secret.generic ins(%0 : !secret.secret<tensor<1024xi16>>) attrs = {__resattrs = [{mgmt.mgmt = #mgmt.mgmt<level = 1>}]} {
    ^body(%input0: tensor<1024xi16>):
      %extracted = tensor.extract %input0[%c0] : tensor<1024xi16>
      secret.yield %extracted : i16
    } -> !secret.secret<i16>
    %2 = secret.generic ins(%1 : !secret.secret<i16>) attrs = {__resattrs = [{mgmt.mgmt = #mgmt.mgmt<level = 0>}]} {
    ^body(%input0: i16):
      %3 = mgmt.modreduce %input0 : i16
      secret.yield %3 : i16
    } -> !secret.secret<i16>
    return %2 : !secret.secret<i16>
  }
}
