// RUN: heir-opt --mlir-print-local-scope --secret-to-bgv %s | FileCheck %s

module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 14, Q = [67239937, 17179967489, 17180262401, 17180295169, 17180393473, 70368744210433], P = [70368744570881, 70368744701953], plaintextModulus = 65537>} {
  func.func @test_arith_ops(
      %arg0: !secret.secret<tensor<1024xi1>> {mgmt.mgmt = #mgmt.mgmt<level = 0>},
      %arg1: !secret.secret<tensor<1024xi1>> {mgmt.mgmt = #mgmt.mgmt<level = 0>},
      %arg2: !secret.secret<tensor<1024xi1>> {mgmt.mgmt = #mgmt.mgmt<level = 0>}
  ) -> (!secret.secret<tensor<1024xi1>> {mgmt.mgmt = #mgmt.mgmt<level = 0, dimension = 3>}) {
    %0 = secret.generic(%arg0: !secret.secret<tensor<1024xi1>>, %arg1: !secret.secret<tensor<1024xi1>>) {
    ^body(%input0: tensor<1024xi1>, %input1: tensor<1024xi1>):
      %2 = arith.addi %input0, %input1 : tensor<1024xi1>
      secret.yield %2 : tensor<1024xi1>
    } -> (!secret.secret<tensor<1024xi1>> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
    %1 = secret.generic(%0: !secret.secret<tensor<1024xi1>>, %arg2: !secret.secret<tensor<1024xi1>>) {
    ^body(%input0: tensor<1024xi1>, %input1: tensor<1024xi1>):
      %2 = arith.muli %input0, %input1 : tensor<1024xi1>
      secret.yield %2 : tensor<1024xi1>
    } -> (!secret.secret<tensor<1024xi1>> {mgmt.mgmt = #mgmt.mgmt<level = 0, dimension = 3>})
    return %1 : !secret.secret<tensor<1024xi1>>
  }
}
