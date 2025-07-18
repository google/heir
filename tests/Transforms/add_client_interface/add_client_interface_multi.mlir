// RUN: heir-opt --add-client-interface=ciphertext-size=1024 %s | FileCheck %s

!ct_ty = !secret.secret<tensor<1024xi16>>

#alignment = #tensor_ext.alignment<in = [32], out = [1024]>
#layout = #tensor_ext.layout<map = (d0) -> (d0), alignment = #alignment>
#original_type = #tensor_ext.original_type<originalType = tensor<32xi16>, layout = #layout>

#scalar_alignment = #tensor_ext.alignment<in = [], out = [1024], insertedDims = [0]>
#scalar_layout = #tensor_ext.layout<map = (d0) -> (d0), alignment = #scalar_alignment>
#scalar_original_type = #tensor_ext.original_type<originalType = i16, layout = #scalar_layout>

func.func @foo(
    %arg0: !ct_ty {tensor_ext.original_type = #original_type}
) -> (!ct_ty {tensor_ext.original_type = #scalar_original_type}) {
  return %arg0 : !ct_ty
}

// CHECK: @foo
// CHECK: @foo__encrypt__arg0
// CHECK-NOT: @foo__encrypt__arg1
// CHECK: @foo__decrypt__

func.func @bar(
    %arg0: !ct_ty {tensor_ext.original_type = #original_type},
    %arg1: !ct_ty {tensor_ext.original_type = #original_type}
) -> (!ct_ty {tensor_ext.original_type = #original_type}) {
  %0 = secret.generic(%arg0: !ct_ty, %arg1: !ct_ty) {
  ^body(%pt_arg0: tensor<1024xi16>, %pt_arg1: tensor<1024xi16>):
    %result = arith.addi %pt_arg0, %pt_arg1 : tensor<1024xi16>
    secret.yield %result : tensor<1024xi16>
  } -> !ct_ty
  return %0 : !ct_ty
}
// CHECK: @bar
// CHECK: @bar__encrypt__arg0
// CHECK: @bar__encrypt__arg1
// CHECK: @bar__decrypt__
