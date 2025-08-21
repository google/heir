// RUN: heir-opt --add-client-interface="ciphertext-size=1024" --canonicalize --cse %s | FileCheck %s

!ct_ty = !secret.secret<tensor<1x1024xi16>>
#layout = #tensor_ext.new_layout<domainSize=1, localSize=5, relation="(d0, d1, d2, d3, d4, d5, d6, d7) : (-d1 + d3 == 0, d0 - d4 * 1024 - d5 == 0, d2 - d6 * 32 - d7 == 0, d5 - d7 == 0, -d0 + 31 >= 0, d0 >= 0, d1 >= 0, d2 >= 0, -d2 + 1023 >= 0, -d0 + d3 * 1024 + 1023 >= 0, d0 - d3 * 1024 >= 0, -d0 + d4 * 1024 + 1023 >= 0, d0 - d4 * 1024 >= 0, -d2 + d6 * 32 + 31 >= 0, d2 - d6 * 32 >= 0)">
#original_type = #tensor_ext.original_type<originalType = tensor<32xi16>, layout = #layout>

// The ISL AST for this layout is:
// for (int c1 = 0; c1 <= 1023; c1 += 1)
//   S(c1 % 32, 0, c1);

// CHECK: func.func @add
// CHECK: func.func @add__encrypt__arg
// CHECK-SAME: %[[arg0:.*]]: tensor<32xi16>
// CHECK-DAG: %[[c1024:.*]] = arith.constant 1024 : index
// CHECK-DAG: %[[c32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECK: %[[v0:.*]] = tensor.empty() : tensor<1x1024xi16>
// CHECK: %[[v1:.*]] = scf.for %[[arg1:.*]] = %[[c0]] to %[[c1024]] step %[[c1]] iter_args(%[[arg2:.*]] = %[[v0]]) -> (tensor<1x1024xi16>) {
// CHECK:  %[[v3:.*]] = arith.remsi %[[arg1]], %[[c32]] : index
// CHECK:  %[[extracted:.*]] = tensor.extract %[[arg0]][%[[v3]]] : tensor<32xi16>
// CHECK:  %[[inserted:.*]] = tensor.insert %[[extracted]] into %[[arg2]][%[[c0]], %[[arg1]]] : tensor<1x1024xi16>
// CHECK:  scf.yield %[[inserted]] : tensor<1x1024xi16>
// CHECK:  }
// CHECK: %[[v2:.*]] = secret.conceal %[[v1]]
// CHECK: return %[[v2]] : !secret.secret<tensor<1x1024xi16>>

func.func @add(
    %arg0: !ct_ty {tensor_ext.original_type = #original_type}
) -> (!ct_ty {tensor_ext.original_type = #original_type}) {
  %0 = secret.generic(%arg0: !ct_ty) {
  ^body(%pt_arg0: tensor<1x1024xi16>):
    %0 = arith.addi %pt_arg0, %pt_arg0 : tensor<1x1024xi16>
    secret.yield %0 : tensor<1x1024xi16>
  } -> !ct_ty
  return %0 : !ct_ty
}
