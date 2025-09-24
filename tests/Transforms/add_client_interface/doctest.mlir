// RUN: heir-opt --add-client-interface=ciphertext-size=1024 %s | FileCheck %s

// CHECK: func.func @simple_add(%[[ARG0:.*]]: !secret.secret<tensor<1x1024xi16>> {{.*}}, %[[ARG1:.*]]: !secret.secret<tensor<1x1024xi16>> {{.*}}) -> (!secret.secret<tensor<1x1024xi16>> {{.*}}) {
// CHECK:   %[[GENERIC:.*]] = secret.generic(%[[ARG0]]: !secret.secret<tensor<1x1024xi16>>, %[[ARG1]]: !secret.secret<tensor<1x1024xi16>>) {
// CHECK:   ^body(%[[PT_ARG0:.*]]: tensor<1x1024xi16>, %[[PT_ARG1:.*]]: tensor<1x1024xi16>):
// CHECK:     %[[ADD:.*]] = arith.addi %[[PT_ARG0]], %[[PT_ARG1]] : tensor<1x1024xi16>
// CHECK:     secret.yield %[[ADD]] : tensor<1x1024xi16>
// CHECK:   return %[[GENERIC]]

// CHECK: func.func @simple_add__encrypt__arg0(%[[CLEAR_ARG0:.*]]: tensor<32xi16>)
// CHECK-SAME: -> !secret.secret<tensor<1x1024xi16>>
// CHECK-SAME: attributes {client.enc_func = {func_name = "simple_add", index = 0 : i64}}

// CHECK: func.func @simple_add__encrypt__arg1(%[[CLEAR_ARG1:.*]]: tensor<32xi16>)
// CHECK-SAME: -> !secret.secret<tensor<1x1024xi16>>
// CHECK-SAME: attributes {client.enc_func = {func_name = "simple_add", index = 1 : i64}}

// CHECK: func.func @simple_add__decrypt__result0(%[[SECRET_RESULT:.*]]: !secret.secret<tensor<1x1024xi16>>)

!ct_ty = !secret.secret<tensor<1x1024xi16>>
#layout = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : (slot - i0) mod 32 = 0 and ct = 0 and 1023 >= slot >= 0 and 31 >= i0 >= 0 }">
#original_type = #tensor_ext.original_type<originalType = tensor<32xi16>, layout = #layout>

func.func @simple_add(
    %arg0: !ct_ty {tensor_ext.original_type = #original_type},
    %arg1: !ct_ty {tensor_ext.original_type = #original_type}
) -> (!ct_ty {tensor_ext.original_type = #original_type}) {
  %0 = secret.generic(%arg0: !ct_ty, %arg1: !ct_ty) {
  ^body(%pt_arg0: tensor<1x1024xi16>, %pt_arg1: tensor<1x1024xi16>):
    %1 = arith.addi %pt_arg0, %pt_arg1 : tensor<1x1024xi16>
    secret.yield %1 : tensor<1x1024xi16>
  } -> !ct_ty
  return %0 : !ct_ty
}
