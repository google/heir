// RUN: heir-opt %s --mlir-print-local-scope --secret-distribute-generic --secret-to-bgv=poly-mod-degree=8 | FileCheck %s

module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 13, Q = [1152921504606994433, 17179967489], P = [1152921504607191041], plaintextModulus = 65537>, scheme.bfv} {
  func.func @two_mul(%arg0: !secret.secret<tensor<8xi16>>, %arg1: !secret.secret<tensor<8xi16>>) -> !secret.secret<tensor<8xi16>> {
    %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>) attrs = {__argattrs = [{mgmt.mgmt = #mgmt.mgmt<level = 1>}, {mgmt.mgmt = #mgmt.mgmt<level = 1>}], __resattrs = [{mgmt.mgmt = #mgmt.mgmt<level = 1>}]} {
    ^body(%input0: tensor<8xi16>, %input1: tensor<8xi16>):
      // CHECK: bgv.mul
      // CHECK-SAME: size = 3
      %1 = arith.muli %input0, %input1 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : tensor<8xi16>
      // CHECK: bgv.mul
      // CHECK-SAME: size = 3
      // CHECK-SAME: size = 4
      %2 = arith.muli %1, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 4>} : tensor<8xi16>
      // CHECK: bgv.relinearize
      // CHECK-SAME: from_basis = array<i32: 0, 1, 2, 3>
      %3 = mgmt.relinearize %2 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<8xi16>
      secret.yield %3 : tensor<8xi16>
    } -> !secret.secret<tensor<8xi16>>
    return %0 : !secret.secret<tensor<8xi16>>
  }
}
