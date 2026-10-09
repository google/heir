// RUN: heir-opt --preprocessing-to-cheddar %s | FileCheck %s --check-prefix=CONVERT
// RUN: heir-opt --preprocessing-to-cheddar --cheddar-bufferize --cse --canonicalize %s | FileCheck %s --check-prefix=BUFFERIZE

// A preprocessing storage of rank-0 Cheddar plaintext tensors becomes a flat
// memref of scalar plaintexts. Move-only payloads are never loaded/stored as
// C++ values: stores constrain the producer to materialize in a rank-0 slot,
// and loads expose a rank-reduced memref view of the storage.

// CONVERT: func @store_cheddar
// CONVERT: %[[STORAGE:.*]] = memref.alloc() : memref<4x!plaintext>
// BUFFERIZE: func @store_cheddar({{.*}}%[[STORAGE:.*]]: memref<4x!plaintext> {bufferize.result})
// BUFFERIZE-NOT: memref.alloc
func.func @store_cheddar(%encoder: !cheddar.encoder, %input0: tensor<4xf64>, %input1: tensor<4xf64>) -> !preprocessing.storage<tensor<!cheddar.plaintext>, tensor<!cheddar.plaintext>> {
  %storage = preprocessing.empty : !preprocessing.storage<tensor<!cheddar.plaintext>, tensor<!cheddar.plaintext>>

  %empty0 = tensor.empty() : tensor<!cheddar.plaintext>
  %arg0 = cheddar.encode %encoder, %input0, %empty0 <level = 1 : i64> : (!cheddar.encoder, tensor<4xf64>, tensor<!cheddar.plaintext>) -> tensor<!cheddar.plaintext>
  // CONVERT: %[[C0:.*]] = arith.constant 0 : index
  // CONVERT: %[[SLOT0:.*]] = memref.subview %[[STORAGE]][%[[C0]]] [1] [1]
  // CONVERT: bufferization.materialize_in_destination %{{.*}} in restrict writable %[[SLOT0]]
  // BUFFERIZE: cheddar.encode %{{.*}}, %{{.*}}, %[[SLOT0:.*]]
  // BUFFERIZE-NOT: memref.copy
  // BUFFERIZE-NOT: memref.store
  preprocessing.store %arg0, %storage[] site 0 <tensor<!cheddar.plaintext>> : tensor<!cheddar.plaintext>, !preprocessing.storage<tensor<!cheddar.plaintext>, tensor<!cheddar.plaintext>>

  %empty1 = tensor.empty() : tensor<!cheddar.plaintext>
  %arg1 = cheddar.encode %encoder, %input1, %empty1 <level = 1 : i64> : (!cheddar.encoder, tensor<4xf64>, tensor<!cheddar.plaintext>) -> tensor<!cheddar.plaintext>
  // CONVERT: %[[C1:.*]] = arith.constant 1 : index
  // CONVERT: %[[SLOT1:.*]] = memref.subview %[[STORAGE]][%[[C1]]] [1] [1]
  // CONVERT: bufferization.materialize_in_destination %{{.*}} in restrict writable %[[SLOT1]]
  preprocessing.store %arg1, %storage[] site 1 <tensor<!cheddar.plaintext>> : tensor<!cheddar.plaintext>, !preprocessing.storage<tensor<!cheddar.plaintext>, tensor<!cheddar.plaintext>>
  return %storage : !preprocessing.storage<tensor<!cheddar.plaintext>, tensor<!cheddar.plaintext>>
}

// CONVERT: func @load_cheddar(%[[CTX:.*]]: !context, %[[CT:.*]]: tensor<!ciphertext>, %[[STORAGE:.*]]: memref<4x!plaintext>)
// CONVERT: %[[LOAD_INDEX:.*]] = arith.constant 0 : index
// CONVERT: %[[PT:.*]] = memref.subview %[[STORAGE]][%[[LOAD_INDEX]]] [1] [1]
// CONVERT: cheddar.mult_plain %[[CTX]], %[[CT]], %[[PT]]
// BUFFERIZE: func @load_cheddar
// BUFFERIZE: %[[PT:.*]] = memref.subview %{{.*}}
// BUFFERIZE: %[[PT_CAST:.*]] = memref.cast %[[PT]]
// BUFFERIZE: cheddar.mult_plain %{{.*}}, %{{.*}}, %[[PT_CAST]]
// BUFFERIZE-NOT: memref.load
func.func @load_cheddar(%ctx: !cheddar.context, %ct: tensor<!cheddar.ciphertext>, %storage: !preprocessing.storage<tensor<!cheddar.plaintext>, tensor<!cheddar.plaintext>>) -> tensor<!cheddar.ciphertext> {
  %pt = preprocessing.load %storage[] site 0 <tensor<!cheddar.plaintext>> : !preprocessing.storage<tensor<!cheddar.plaintext>, tensor<!cheddar.plaintext>>, tensor<!cheddar.plaintext>
  %out = tensor.empty() : tensor<!cheddar.ciphertext>
  %result = cheddar.mult_plain %ctx, %ct, %pt, %out : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.plaintext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// A loop-indexed preprocessing site writes each produced plaintext directly
// into its dynamic storage slot; no move-only copy is introduced.
// BUFFERIZE: func.func @store_cheddar_loop
// BUFFERIZE: scf.for %[[I:[^ ]+]]
// BUFFERIZE: %[[INDEX:.*]] = arith.addi %[[I]], %{{.*}} : index
// BUFFERIZE: %[[SLOT:.*]] = memref.subview %{{.*}}[%[[INDEX]]] [1] [1]
// BUFFERIZE: cheddar.encode %{{.*}}, %{{.*}}, %[[SLOT]]
// BUFFERIZE-NOT: memref.copy
func.func @store_cheddar_loop(%encoder: !cheddar.encoder, %input: tensor<4xf64>) -> !preprocessing.storage<tensor<!cheddar.plaintext>> {
  %storage = preprocessing.empty : !preprocessing.storage<tensor<!cheddar.plaintext>>
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c2 step %c1 {
    %empty = tensor.empty() : tensor<!cheddar.plaintext>
    %encoded = cheddar.encode %encoder, %input, %empty <level = 1 : i64> : (!cheddar.encoder, tensor<4xf64>, tensor<!cheddar.plaintext>) -> tensor<!cheddar.plaintext>
    preprocessing.store %encoded, %storage[%i] site 2 <tensor<!cheddar.plaintext>> : tensor<!cheddar.plaintext>, !preprocessing.storage<tensor<!cheddar.plaintext>>
  }
  return %storage : !preprocessing.storage<tensor<!cheddar.plaintext>>
}

// A module containing only generic external resources needs no Cheddar
// storage conversion and remains legal for the later target lowering.
// CONVERT: func.func @resource_only
// CONVERT: preprocessing.load_resource "weights.bin"
// BUFFERIZE: func.func @resource_only
// BUFFERIZE: preprocessing.load_resource "weights.bin"
func.func @resource_only() -> tensor<4xf32> {
  %destination = tensor.empty() : tensor<4xf32>
  %resource = preprocessing.load_resource "weights.bin" into %destination
      : (tensor<4xf32>) -> tensor<4xf32>
  return %resource : tensor<4xf32>
}
