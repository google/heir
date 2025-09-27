// RUN: heir-opt --add-client-interface %s | FileCheck %s

!ct_ty = !secret.secret<tensor<1x1024xi16>>
#layout = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : (slot - i0) mod 32 = 0 and ct = 0 and 1023 >= slot >= 0 and 31 >= i0 >= 0 }">
#original_type = #tensor_ext.original_type<originalType = tensor<32xi16>, layout = #layout>

func.func @add(
    %arg0: !ct_ty {tensor_ext.original_type = #original_type},
    %arg1: !ct_ty {tensor_ext.original_type = #original_type}
) -> (!ct_ty {tensor_ext.original_type = #original_type}, !ct_ty {tensor_ext.original_type = #original_type}) {
  %0 = secret.generic(%arg0: !ct_ty, %arg1: !ct_ty) {
  ^body(%pt_arg0: tensor<1x1024xi16>, %pt_arg1: tensor<1x1024xi16>):
    %0 = arith.addi %pt_arg0, %pt_arg1 : tensor<1x1024xi16>
    secret.yield %0 : tensor<1x1024xi16>
  } -> !ct_ty
  return %0, %0 : !ct_ty, !ct_ty
}

// CHECK: func.func @add__encrypt__arg0
// CHECK: func.func @add__encrypt__arg1
// CHECK: func.func @add__decrypt__result0
// CHECK: func.func @add__decrypt__result1
