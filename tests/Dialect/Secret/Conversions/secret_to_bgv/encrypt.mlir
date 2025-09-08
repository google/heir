// RUN: heir-opt --secret-to-bgv="poly-mod-degree=1024" %s | FileCheck %s

#original_type = #tensor_ext.original_type<originalType = tensor<1024xi16>, layout = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 1024 = 0 and 0 <= i0 <= 1023 and 0 <= slot <= 1023 }">>

module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 14, Q = [67239937, 17179967489, 17180262401, 17180295169, 17180393473, 70368744210433], P = [70368744570881, 70368744701953], plaintextModulus = 65537>} {

  // A dummy placeholder function
  func.func @hamming(%arg0: !secret.secret<tensor<1x1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 5, scale = 1>, tensor_ext.original_type = #original_type}) -> (!secret.secret<tensor<1x1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 5, scale = 1>, tensor_ext.original_type = #original_type}) {
    return %arg0: !secret.secret<tensor<1x1024xi16>>
  }

  func.func @hamming__encrypt__arg0(%arg0: tensor<1024xi16>) -> (!secret.secret<tensor<1x1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 5, scale = 1>}) attributes {client.enc_func = {func_name = "hamming", index = 0 : i64}} {
    %cst = arith.constant dense<0> : tensor<1x1024xi16>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %0 = scf.for %arg1 = %c0 to %c1024 step %c1 iter_args(%arg2 = %cst) -> (tensor<1x1024xi16>) {
      %extracted = tensor.extract %arg0[%arg1] : tensor<1024xi16>
      %inserted = tensor.insert %extracted into %arg2[%c0, %arg1] : tensor<1x1024xi16>
      scf.yield %inserted : tensor<1x1024xi16>
    }
    // CHECK-NOT: secret.conceal
    %1 = secret.conceal %0 {mgmt.mgmt = #mgmt.mgmt<level = 5, scale = 1>} : tensor<1x1024xi16> -> !secret.secret<tensor<1x1024xi16>>
    return %1 : !secret.secret<tensor<1x1024xi16>>
  }
}
