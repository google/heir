// RUN: heir-opt --secret-to-bgv="poly-mod-degree=8" --verify-diagnostics %s

// This example was converted from the dot_product_8f.mlir example via
//
//   heir-opt --mlir-print-ir-before-all --mlir-to-bgv='ciphertext-degree=8' --scheme-to-openfhe='entry-function=dot_product' tests/Examples/openfhe/dot_product_8f.mlir

module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 14, Q = [67239937, 17179967489, 17180262401, 17180295169, 17180393473, 70368744210433], P = [70368744570881, 70368744701953], plaintextModulus = 65537>} {
  func.func @dot_product(%arg0: !secret.secret<tensor<8xf16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>}, %arg1: !secret.secret<tensor<8xf16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>}) -> (!secret.secret<tensor<8xf16>> {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>}) {
    %cst = arith.constant dense<9.997550e-02> : tensor<8xf16>
    // expected-error@below {{Floating point types are not supported in BGV. Maybe you meant to use a CKKS pipeline like --mlir-to-ckks?}}
    %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<8xf16>>, !secret.secret<tensor<8xf16>>) attrs = {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} {
    ^bb0(%arg2: tensor<8xf16>, %arg3: tensor<8xf16>):
      %17 = arith.mulf %arg2, %arg3 : tensor<8xf16>
      secret.yield %17 : tensor<8xf16>
    } -> !secret.secret<tensor<8xf16>>
    %1 = secret.generic ins(%0 : !secret.secret<tensor<8xf16>>) attrs = {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>} {
    ^bb0(%arg2: tensor<8xf16>):
      %17 = arith.addf %arg2, %cst : tensor<8xf16>
      secret.yield %17 : tensor<8xf16>
    } -> !secret.secret<tensor<8xf16>>
    return %1 : !secret.secret<tensor<8xf16>>
  }
}
