// RUN: heir-opt --secret-to-bgv %s | FileCheck %s

!ct = !secret.secret<tensor<1024xi1>>
#mgmt = #mgmt.mgmt<level = 0, dimension = 2>

module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 14, Q = [67239937, 17179967489, 17180262401, 17180295169, 17180393473, 70368744210433], P = [70368744570881, 70368744701953], plaintextModulus = 65537>} {
  // CHECK: func @test_arith_ops
  func.func @test_arith_ops(%arg0 : tensor<1024xi1>) -> (!ct {mgmt.mgmt = #mgmt}) {
    // CHECK: lwe.rlwe_encode
    // CHECK: lwe.rlwe_trivial_encrypt
    %0 = secret.conceal %arg0 {trivial, mgmt.mgmt = #mgmt} : tensor<1024xi1> -> !ct
    // CHECK: return
    return %0 : !ct
  }
}
