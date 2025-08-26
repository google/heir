// RUN: heir-opt --add-client-interface="ciphertext-size=1024" --canonicalize --cse %s | FileCheck %s
!ct_ty = !secret.secret<tensor<16x1024xi16>>
#layout = #tensor_ext.new_layout<"{ [i0, i1] -> [ct, slot] : (i0 - i1 + ct) mod 16 = 0 and (-i0 + slot) mod 16 = 0 and 0 <= i0 <= 9 and 0 <= i1 <= 15 and 0 <= ct <= 15 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = tensor<10x16xi16>, layout = #layout>

// CHECK: func.func @add
// CHECK: func.func @add__encrypt__arg
// CHECK-DAG: %[[c1024:.*]] = arith.constant 1024 : index
// CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
// CHECK: scf.for %[[arg1:.*]] = %[[c0]] to %[[c1024]] step %[[c1]]
// CHECK: scf.if
// CHECK: return
func.func @add(
    %arg0: !ct_ty {tensor_ext.original_type = #original_type}
) -> (!ct_ty {tensor_ext.original_type = #original_type}) {
  %0 = secret.generic(%arg0: !ct_ty) {
  ^body(%pt_arg0: tensor<16x1024xi16>):
    %0 = arith.addi %pt_arg0, %pt_arg0 : tensor<16x1024xi16>
    secret.yield %0 : tensor<16x1024xi16>
  } -> !ct_ty
  return %0 : !ct_ty
}
