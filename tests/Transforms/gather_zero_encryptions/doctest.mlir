// RUN: heir-opt --gather-zero-encryptions %s | FileCheck %s

!ct = !openfhe.ciphertext
!cc = !openfhe.crypto_context
!pk = !openfhe.public_key
!pt = !openfhe.plaintext

// CHECK: func.func @compute
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ZEROS:.*]]: memref<2x!ct> {client.enc_zero_arg}) -> (!ct, !ct)
func.func @compute(%arg0: i32, %z0: !ct {client.enc_zero_arg = {index = 0 : i64}}, %z1: !ct {client.enc_zero_arg = {index = 1 : i64}}) -> (!ct, !ct) {
  return %z0, %z1 : !ct, !ct
}

// CHECK-NOT: func.func @compute__encrypt__zero__0
// CHECK: func.func @compute__encrypt__zeros
// CHECK-SAME: (%[[CC:.*]]: !cc, %[[PK:.*]]: !pk) -> memref<2x!ct> attributes {client.enc_zero_func = {func_name = "compute"}}

func.func @compute__encrypt__zero__0(%cc: !cc, %pk: !pk) -> !ct attributes {client.enc_zero_func = {func_name = "compute", index = 0 : i64}} {
  %cst = arith.constant dense<0> : tensor<16xi32>
  %pt = openfhe.make_packed_plaintext %cc, %cst : (!cc, tensor<16xi32>) -> !pt
  %0 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
  return %0 : !ct
}

func.func @compute__encrypt__zero__1(%cc: !cc, %pk: !pk) -> !ct attributes {client.enc_zero_func = {func_name = "compute", index = 1 : i64}} {
  %cst = arith.constant dense<0> : tensor<1xi32>
  %pt = openfhe.make_packed_plaintext %cc, %cst : (!cc, tensor<1xi32>) -> !pt
  %1 = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct
  return %1 : !ct
}
