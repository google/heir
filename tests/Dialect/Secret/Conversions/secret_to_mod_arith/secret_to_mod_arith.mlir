// RUN: heir-opt --secret-distribute-generic --secret-to-mod-arith=modulus=17 %s | FileCheck %s

// CHECK-NOT: secret.generic
#alignment = #tensor_ext.alignment<in = [], out = [1024], insertedDims = [0]>
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 mod 1024)>
#layout = #tensor_ext.layout<map = (d0) -> (d0 mod 1024), alignment = #alignment>
#original_type = #tensor_ext.original_type<originalType = i16, layout = #layout>
module {
  func.func @dot_product(%arg0: !secret.secret<tensor<1024xi16>> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<8xi16>, layout = <map = (d0) -> (d0 mod 1024), alignment = <in = [8], out = [1024]>>>}, %arg1: !secret.secret<tensor<1024xi16>> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<8xi16>, layout = <map = (d0) -> (d0 mod 1024), alignment = <in = [8], out = [1024]>>>}) -> (!secret.secret<tensor<1024xi16>> {tensor_ext.original_type = #original_type}) {
    %c7 = arith.constant 7 : index
    %c1_i16 = arith.constant 1 : i16
    %cst = arith.constant dense<0> : tensor<1024xi16>
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %inserted = tensor.insert %c1_i16 into %cst[%c7] : tensor<1024xi16>
    %0 = secret.generic(%arg0: !secret.secret<tensor<1024xi16>>, %arg1: !secret.secret<tensor<1024xi16>>) {
    ^body(%input0: tensor<1024xi16>, %input1: tensor<1024xi16>):
      %1 = arith.muli %input0, %input1 : tensor<1024xi16>
      %2 = tensor_ext.rotate %1, %c4 : tensor<1024xi16>, index
      %3 = arith.addi %1, %2 : tensor<1024xi16>
      %4 = tensor_ext.rotate %3, %c2 : tensor<1024xi16>, index
      %5 = arith.addi %3, %4 : tensor<1024xi16>
      %6 = tensor_ext.rotate %5, %c1 : tensor<1024xi16>, index
      %7 = arith.addi %5, %6 : tensor<1024xi16>
      %8 = arith.muli %inserted, %7 : tensor<1024xi16>
      %9 = tensor_ext.rotate %8, %c7 : tensor<1024xi16>, index
      secret.yield %9 : tensor<1024xi16>
    } -> !secret.secret<tensor<1024xi16>>
    return %0 : !secret.secret<tensor<1024xi16>>
  }
  func.func @dot_product__encrypt__arg0(%arg0: tensor<8xi16>) -> !secret.secret<tensor<1024xi16>> attributes {client.enc_func = {func_name = "dot_product", index = 0 : i64}} {
    %cst = arith.constant dense<0> : tensor<1024xi16>
    %concat = tensor.concat dim(0) %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0 : (tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>) -> tensor<1024xi16>
    %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%concat : tensor<1024xi16>) outs(%cst : tensor<1024xi16>) {
    ^bb0(%in: i16, %out: i16):
      linalg.yield %in : i16
    } -> tensor<1024xi16>
    %1 = secret.conceal %0 : tensor<1024xi16> -> !secret.secret<tensor<1024xi16>>
    return %1 : !secret.secret<tensor<1024xi16>>
  }
  func.func @dot_product__encrypt__arg1(%arg0: tensor<8xi16>) -> !secret.secret<tensor<1024xi16>> attributes {client.enc_func = {func_name = "dot_product", index = 1 : i64}} {
    %cst = arith.constant dense<0> : tensor<1024xi16>
    %concat = tensor.concat dim(0) %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0 : (tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>, tensor<8xi16>) -> tensor<1024xi16>
    %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%concat : tensor<1024xi16>) outs(%cst : tensor<1024xi16>) {
    ^bb0(%in: i16, %out: i16):
      linalg.yield %in : i16
    } -> tensor<1024xi16>
    %1 = secret.conceal %0 : tensor<1024xi16> -> !secret.secret<tensor<1024xi16>>
    return %1 : !secret.secret<tensor<1024xi16>>
  }
  func.func @dot_product__decrypt__result0(%arg0: !secret.secret<tensor<1024xi16>>) -> i16 attributes {client.dec_func = {func_name = "dot_product", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %0 = secret.reveal %arg0 : !secret.secret<tensor<1024xi16>> -> tensor<1024xi16>
    %extracted = tensor.extract %0[%c0] : tensor<1024xi16>
    return %extracted : i16
  }
}
