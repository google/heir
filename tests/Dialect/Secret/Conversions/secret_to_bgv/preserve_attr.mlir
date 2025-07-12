// RUN: heir-opt --secret-to-bgv %s | FileCheck %s

!eui1 = !secret.secret<tensor<1024xi1>>
#mgmt = #mgmt.mgmt<level = 0, dimension = 2>

module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 14, Q = [67239937, 17179967489, 17180262401, 17180295169, 17180393473, 70368744210433], P = [70368744570881, 70368744701953], plaintextModulus = 65537>} {
  // CHECK: func @test_preserve_attr
  func.func @test_preserve_attr(%arg0 : !eui1 {mgmt.mgmt = #mgmt}, %arg1 : !eui1 {mgmt.mgmt = #mgmt}, %arg2 : !eui1 {mgmt.mgmt = #mgmt}) -> (!eui1 {mgmt.mgmt = #mgmt}) {
    %0 = secret.generic(%arg0: !eui1, %arg1: !eui1) {
    // CHECK: {dialect.attr = 1 : i64}
      ^bb0(%ARG0 : tensor<1024xi1>, %ARG1 : tensor<1024xi1>):
        %1 = arith.addi %ARG0, %ARG1 {dialect.attr = 1} : tensor<1024xi1>
        secret.yield %1 : tensor<1024xi1>
    } -> (!eui1 {mgmt.mgmt = #mgmt})
    return %0 : !eui1
  }
}
