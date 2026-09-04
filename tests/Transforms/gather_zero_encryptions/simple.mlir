// RUN: heir-opt --gather-zero-encryptions %s | FileCheck %s

!ct = !openfhe.ciphertext
!cc = !openfhe.crypto_context
!pk = !openfhe.public_key

// CHECK: func.func @compute
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ZEROS:.*]]: memref<2x!ct> {client.enc_zero_arg}) -> (!ct, !ct)
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[LOAD0:.*]] = memref.load %[[ZEROS]][%[[C0]]] : memref<2x!ct>
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[LOAD1:.*]] = memref.load %[[ZEROS]][%[[C1]]] : memref<2x!ct>
// CHECK: return %[[LOAD0]], %[[LOAD1]] : !ct, !ct
func.func @compute(%arg0: i32, %z0: !ct {client.enc_zero_arg = {index = 0 : i64}}, %z1: !ct {client.enc_zero_arg = {index = 1 : i64}}) -> (!ct, !ct) {
  return %z0, %z1 : !ct, !ct
}

// CHECK-NOT: func.func @compute__encrypt__zero__0
// CHECK-NOT: func.func @compute__encrypt__zero__1

// CHECK: func.func @compute__encrypt__zeros
// CHECK-SAME: (%[[CC:.*]]: !cc, %[[PK:.*]]: !pk) -> memref<2x!ct> attributes {client.enc_zero_func = {func_name = "compute"}}
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc() : memref<2x!ct>
// CHECK: %[[ZERO0:.*]] = builtin.unrealized_conversion_cast
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: memref.store %[[ZERO0]], %[[ALLOC]][%[[C0]]] : memref<2x!ct>
// CHECK: %[[ZERO1:.*]] = builtin.unrealized_conversion_cast
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: memref.store %[[ZERO1]], %[[ALLOC]][%[[C1]]] : memref<2x!ct>
// CHECK: return %[[ALLOC]] : memref<2x!ct>

func.func @compute__encrypt__zero__0(%cc: !cc, %pk: !pk) -> !ct attributes {client.enc_zero_func = {func_name = "compute", index = 0 : i64}} {
  %0 = builtin.unrealized_conversion_cast %cc : !cc to !ct
  return %0 : !ct
}

func.func @compute__encrypt__zero__1(%cc: !cc, %pk: !pk) -> !ct attributes {client.enc_zero_func = {func_name = "compute", index = 1 : i64}} {
  %1 = builtin.unrealized_conversion_cast %cc : !cc to !ct
  return %1 : !ct
}
