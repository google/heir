// RUN: heir-opt %s --convert-to-ciphertext-semantics=ciphertext-size=16 | FileCheck %s

// Tensor is repeated twice, so the packed cleartext should use two nonzero slots
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (slot - i0) mod 8 = 0 and 0 <= i0 <= 7 and 0 <= slot <= 15 }">
// Scalar is repeated throughout the ciphertext
#scalar_layout = #tensor_ext.layout<"{ [] -> [ct, slot] : ct = 0 and 0 <= slot <= 15 }">

// CHECK: func.func @extract_static_indices
// CHECK: [[C0:%.+]] = arith.constant 0 : index
// CHECK: [[C1:%.+]] = arith.constant 1 : i16
// CHECK: [[CST:%.+]] = arith.constant dense<0> : tensor<1x16xi16>
// CHECK: [[C3:%.+]] = arith.constant 3 : index
// CHECK: [[RESULT:%.+]] = secret.generic([[ARG0:%.+]]: !secret.secret<tensor<1x16xi16>>)
// CHECK: ^body([[ARG0_INNER:%.+]]: tensor<1x16xi16>):
// CHECK: [[INSERTED:%.+]] = tensor.insert [[C1]] into [[CST]]{{\[}}[[C0]], [[C3]]] : tensor<1x16xi16>
// CHECK: [[MUL:%.+]] = arith.muli [[INSERTED]], [[ARG0_INNER]] : tensor<1x16xi16>
// CHECK: [[PERMUTED:%.+]] = tensor_ext.permute [[MUL]] {permutation = {{.*}}} : tensor<1x16xi16>
// CHECK: secret.yield [[PERMUTED]] : tensor<1x16xi16>
// CHECK: return [[RESULT]] : !secret.secret<tensor<1x16xi16>>

func.func @extract_static_indices(%arg0: !secret.secret<tensor<8xi16>> {tensor_ext.layout = #layout})
      -> (!secret.secret<i16> {tensor_ext.layout = #scalar_layout}) {
  %index = arith.constant 3 : index
  %0 = secret.generic(%arg0 : !secret.secret<tensor<8xi16>> {tensor_ext.layout = #layout}) {
  ^body(%input0: tensor<8xi16>):
    %0 = tensor.extract %input0[%index] {tensor_ext.layout = #scalar_layout} : tensor<8xi16>
    secret.yield %0 : i16
  } -> (!secret.secret<i16> {tensor_ext.layout = #layout})
  return %0 : !secret.secret<i16>
}
