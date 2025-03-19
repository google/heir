// RUN: heir-opt --mlir-print-local-scope --secret-to-bgv=poly-mod-degree=8 %s | FileCheck %s

#alignment = #tensor_ext.alignment<in = [], out = [8], insertedDims = [0]>
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 mod 8)>
#layout = #tensor_ext.layout<map = (d0) -> (d0 mod 8), alignment = #alignment>
#original_type = #tensor_ext.original_type<originalType = i16, layout = #layout>
module attributes {backend.openfhe, bgv.schemeParam = #bgv.scheme_param<logN = 14, Q = [35184372121601, 35184372744193, 35184373006337], P = [35184373989377, 35184374874113], plaintextModulus = 65537>, scheme.bgv} {
  func.func @dot_product(%arg0: !secret.secret<tensor<8xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>, tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<8xi16>, layout = <map = (d0) -> (d0 mod 8)>>}, %arg1: !secret.secret<tensor<8xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>, tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<8xi16>, layout = <map = (d0) -> (d0 mod 8)>>}) -> (!secret.secret<tensor<8xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>, tensor_ext.original_type = #original_type}) attributes {mgmt.openfhe_params = #mgmt.openfhe_params<evalAddCount = 8, keySwitchCount = 15>} {
    %1 = secret.generic(%arg0: !secret.secret<tensor<8xi16>>, %arg1: !secret.secret<tensor<8xi16>>) {
    ^body(%input0: tensor<8xi16>, %input1: tensor<8xi16>):
      %11 = arith.muli %input0, %input1 : tensor<8xi16>
      secret.yield %11 : tensor<8xi16>
    } -> (!secret.secret<tensor<8xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2, dimension = 3>})
    %2 = secret.generic(%1: !secret.secret<tensor<8xi16>>) {
    ^body(%input0: tensor<8xi16>):
      %11 = mgmt.relinearize %input0 : tensor<8xi16>
      secret.yield %11 : tensor<8xi16>
    } -> (!secret.secret<tensor<8xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>})
    return %2 : !secret.secret<tensor<8xi16>>
  }

  // CHECK: @dot_product__encrypt__arg0
  // CHECK-SAME: !lwe.new_lwe_public_key
  // CHECK: lwe.rlwe_encode
  // CHECK: lwe.rlwe_encrypt
  func.func @dot_product__encrypt__arg0(%arg0: tensor<8xi16>) -> (!secret.secret<tensor<8xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>}) attributes {client.enc_func = {func_name = "dot_product", index = 0 : i64}} {
    %cst = arith.constant dense<0> : tensor<8xi16>
    %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%arg0 : tensor<8xi16>) outs(%cst : tensor<8xi16>) {
    ^bb0(%in: i16, %out: i16):
      linalg.yield %in : i16
    } -> tensor<8xi16>
    %1 = secret.conceal %0 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16> -> !secret.secret<tensor<8xi16>>
    return %1 : !secret.secret<tensor<8xi16>>
  }

  // CHECK: @dot_product__encrypt__arg1
  // CHECK-SAME: !lwe.new_lwe_public_key
  // CHECK: lwe.rlwe_encode
  // CHECK: lwe.rlwe_encrypt
  func.func @dot_product__encrypt__arg1(%arg0: tensor<8xi16>) -> (!secret.secret<tensor<8xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>}) attributes {client.enc_func = {func_name = "dot_product", index = 1 : i64}} {
    %cst = arith.constant dense<0> : tensor<8xi16>
    %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%arg0 : tensor<8xi16>) outs(%cst : tensor<8xi16>) {
    ^bb0(%in: i16, %out: i16):
      linalg.yield %in : i16
    } -> tensor<8xi16>
    %1 = secret.conceal %0 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<8xi16> -> !secret.secret<tensor<8xi16>>
    return %1 : !secret.secret<tensor<8xi16>>
  }

  // CHECK: @dot_product__decrypt__result0
  // CHECK-SAME: !lwe.new_lwe_secret_key
  // CHECK: lwe.rlwe_decrypt
  // CHECK: lwe.rlwe_decode
  func.func @dot_product__decrypt__result0(%arg0: !secret.secret<tensor<8xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>}) -> i16 attributes {client.dec_func = {func_name = "dot_product", index = 0 : i64}} {
    %c0 = arith.constant 0 : index
    %0 = secret.reveal %arg0 : !secret.secret<tensor<8xi16>> -> tensor<8xi16>
    %extracted = tensor.extract %0[%c0] : tensor<8xi16>
    return %extracted : i16
  }
}
