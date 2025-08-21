// RUN: heir-opt --add-client-interface="ciphertext-size=32" --canonicalize --cse %s | FileCheck %s

// Go from 16xi16 -> 1x32xi16 with a layout that strides the input by 2
!ct_ty = !secret.secret<tensor<1x32xi16>>
#layout = #tensor_ext.new_layout<domainSize=1, relation="(d0, d1, d2) : (d2 - 2 * d0 == 0, d0 >= 0, d2 >= 0, 15 >= d0, 31 >= d2, d1 == 0)">
#original_type = #tensor_ext.original_type<originalType = tensor<16xi16>, layout = #layout>

// The ISL AST for this layout is:
// for (int c1 = 0; c1 <= 30; c1 += 2)
//   S(c1 / 2, 0, c1);
//
// Note that the UB of the loop is 31 since the last slot isn't populated with data.

// CHECK: func.func @add
// CHECK: func.func @add__encrypt__arg
// CHECK-DAG: %[[c31:.*]] = arith.constant 31 : index
// CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
// CHECK: scf.for %[[arg1:.*]] = %[[c0]] to %[[c31]] step %[[c2]]
// CHECK: arith.divsi %[[arg1]], %[[c2]]
// CHECK: return

func.func @add(
    %arg0: !ct_ty {tensor_ext.original_type = #original_type}
) -> (!ct_ty {tensor_ext.original_type = #original_type}) {
  %0 = secret.generic(%arg0: !ct_ty) {
  ^body(%pt_arg0: tensor<1x32xi16>):
    %0 = arith.addi %pt_arg0, %pt_arg0 : tensor<1x32xi16>
    secret.yield %0 : tensor<1x32xi16>
  } -> !ct_ty
  return %0 : !ct_ty
}
