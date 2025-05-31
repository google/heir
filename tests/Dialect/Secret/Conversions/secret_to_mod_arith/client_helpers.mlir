// RUN: heir-opt --secret-distribute-generic --secret-to-mod-arith=modulus=17 %s | FileCheck %s

// CHECK-NOT: secret.generic
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 mod 1024)>
module {
  // CHECK: @dot_product__encrypt__arg0
  func.func @dot_product__encrypt__arg0(%arg0: tensor<8xi16>) -> !secret.secret<tensor<1024xi16>> attributes {client.enc_func = {func_name = "dot_product", index = 0 : i64}} {
    %cst = arith.constant dense<0> : tensor<1024xi16>
    %concat = tensor.concat dim(0) %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0 : (tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>) -> tensor<1024xi16>
    %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%concat : tensor<1024xi16>) outs(%cst : tensor<1024xi16>) {
    ^bb0(%in: i16, %out: i16):
      linalg.yield %in : i16
    } -> tensor<1024xi16>
    // CHECK: arith.extsi
    // CHECK: mod_arith.encapsulate
    %1 = secret.conceal %0 : tensor<1024xi16> -> !secret.secret<tensor<1024xi16>>
    return %1 : !secret.secret<tensor<1024xi16>>
  }
  // CHECK: @dot_product__decrypt__result0
  func.func @dot_product__decrypt__result0(%arg0: !secret.secret<tensor<1024xi16>>) -> i16 attributes {client.dec_func = {func_name = "dot_product", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    // CHECK: mod_arith.extract
    // CHECK: arith.trunci
    %0 = secret.reveal %arg0 : !secret.secret<tensor<1024xi16>> -> tensor<1024xi16>
    %extracted = tensor.extract %0[%c0] : tensor<1024xi16>
    return %extracted : i16
  }
}
