// RUN: heir-opt --add-client-interface="ciphertext-size=1024" --canonicalize %s | FileCheck %s

// Data is 32x64, being packed into ciphertexts of size 1024 via Halevi-Shoup
// diagonal layout.
!ct_ty = !secret.secret<tensor<32x1024xi16>>
#layout = #tensor_ext.new_layout<"{ [row, col] -> [ct, slot] : (slot mod 32) - row = 0 and (ct + slot) mod 64 - col = 0 and row >= 0 and col >= 0 and ct >= 0 and slot >= 0 and 1023 - slot >= 0 and 31 - ct >= 0 and 31 - row >= 0 and 64 - col >= 0 }">
#original_type = #tensor_ext.original_type<originalType = tensor<32x64xi16>, layout = #layout>

// The ISL generated loop nest is:
// for (int c0 = 0; c0 <= 31; c0 += 1)
//   for (int c1 = 0; c1 <= 1023; c1 += 1)
//     S(c1 % 32, -((-c0 - c1 + 1087) % 64) + 63, c0, c1);

// CHECK: func.func @add

// CHECK: func.func @add__encrypt__arg
// CHECK-SAME: %[[arg0:.*]]: tensor<32x64xi16>
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[c32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[c1024:.*]] = arith.constant 1024 : index
// CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECK: %[[v0:.*]] = tensor.empty() : tensor<32x1024xi16>
// CHECK: %[[v1:.*]] = scf.for %[[arg1:.*]] = %[[c0]] to %[[c32]] step %[[c1]]
// CHECK:   %[[v3:.*]] = scf.for %[[arg3:.*]] = %[[c0]] to %[[c1024]] step %[[c1]]
// CHECK-COUNT-2:          arith.remsi
// CHECK: %[[v2:.*]] = secret.conceal %[[v1]]
// CHECK: return %[[v2]] : !secret.secret<tensor<32x1024xi16>>

func.func @add(
    %arg0: !ct_ty {tensor_ext.original_type = #original_type}
) -> (!ct_ty {tensor_ext.original_type = #original_type}) {
  %0 = secret.generic(%arg0: !ct_ty) {
  ^body(%pt_arg0: tensor<32x1024xi16>):
    %0 = arith.addi %pt_arg0, %pt_arg0 : tensor<32x1024xi16>
    secret.yield %0 : tensor<32x1024xi16>
  } -> !ct_ty
  return %0 : !ct_ty
}
