// RUN: heir-opt --preprocessing-to-lattigo --split-input-file %s | FileCheck %s

// CHECK: ![[PT:.*]] = !lattigo.rlwe.plaintext

// CHECK: func @test_lattigo
// CHECK-SAME: (%[[arg0:.*]]: ![[PT]]) -> ![[PT]]
func.func @test_lattigo(%arg0: !lattigo.rlwe.plaintext) -> !lattigo.rlwe.plaintext {
  // CHECK: %[[storage:.*]] = memref.alloc() : memref<2x![[PT]]>
  %storage = preprocessing.empty : !preprocessing.storage<!lattigo.rlwe.plaintext, !lattigo.rlwe.plaintext>

  // CHECK: %[[c0:.*]] = arith.constant 0 : index
  // CHECK: memref.store %[[arg0]], %[[storage]][%[[c0]]] : memref<2x![[PT]]>
  preprocessing.store %arg0, %storage[] site 0 <!lattigo.rlwe.plaintext> : !lattigo.rlwe.plaintext, !preprocessing.storage<!lattigo.rlwe.plaintext, !lattigo.rlwe.plaintext>

  // CHECK: %[[c1:.*]] = arith.constant 1 : index
  // CHECK: memref.store %[[arg0]], %[[storage]][%[[c1]]] : memref<2x![[PT]]>
  preprocessing.store %arg0, %storage[] site 1 <!lattigo.rlwe.plaintext> : !lattigo.rlwe.plaintext, !preprocessing.storage<!lattigo.rlwe.plaintext, !lattigo.rlwe.plaintext>

  // CHECK: %[[c0_1:.*]] = arith.constant 0 : index
  // CHECK: %[[res:.*]] = memref.load %[[storage]][%[[c0_1]]] : memref<2x![[PT]]>
  %res = preprocessing.load %storage[] site 0 <!lattigo.rlwe.plaintext> : !preprocessing.storage<!lattigo.rlwe.plaintext, !lattigo.rlwe.plaintext>, !lattigo.rlwe.plaintext
  return %res : !lattigo.rlwe.plaintext
}

// -----

// A storage holding more than one element type -- plaintexts next to prepared
// linear transformations -- becomes one memref per type, with each site
// indexing the memref of its own type.

// CHECK-DAG: ![[LT:.*]] = !lattigo.ckks.linear_transformation
// CHECK-DAG: ![[PT2:.*]] = !lattigo.rlwe.plaintext

// CHECK: func @test_lattigo_multi_type
// CHECK-SAME: (%[[pt:.*]]: ![[PT2]], %[[lt:.*]]: ![[LT]]) -> ![[LT]]
func.func @test_lattigo_multi_type(
    %pt: !lattigo.rlwe.plaintext,
    %lt: !lattigo.ckks.linear_transformation) -> !lattigo.ckks.linear_transformation {
  // CHECK-DAG: %[[storage_pt:.*]] = memref.alloc() : memref<1x![[PT2]]>
  // CHECK-DAG: %[[storage_lt:.*]] = memref.alloc() : memref<2x![[LT]]>
  %storage = preprocessing.empty : !preprocessing.storage<!lattigo.rlwe.plaintext, !lattigo.ckks.linear_transformation, !lattigo.ckks.linear_transformation>

  // The plaintext site indexes the plaintext memref at 0, not the flat site
  // number, so it does not collide with the transformations.
  // CHECK: %[[c0:.*]] = arith.constant 0 : index
  // CHECK: memref.store %[[pt]], %[[storage_pt]][%[[c0]]] : memref<1x![[PT2]]>
  preprocessing.store %pt, %storage[] site 0 <!lattigo.rlwe.plaintext> : !lattigo.rlwe.plaintext, !preprocessing.storage<!lattigo.rlwe.plaintext, !lattigo.ckks.linear_transformation, !lattigo.ckks.linear_transformation>

  // CHECK: %[[c0_0:.*]] = arith.constant 0 : index
  // CHECK: memref.store %[[lt]], %[[storage_lt]][%[[c0_0]]] : memref<2x![[LT]]>
  preprocessing.store %lt, %storage[] site 1 <!lattigo.ckks.linear_transformation> : !lattigo.ckks.linear_transformation, !preprocessing.storage<!lattigo.rlwe.plaintext, !lattigo.ckks.linear_transformation, !lattigo.ckks.linear_transformation>

  // CHECK: %[[c1:.*]] = arith.constant 1 : index
  // CHECK: memref.store %[[lt]], %[[storage_lt]][%[[c1]]] : memref<2x![[LT]]>
  preprocessing.store %lt, %storage[] site 2 <!lattigo.ckks.linear_transformation> : !lattigo.ckks.linear_transformation, !preprocessing.storage<!lattigo.rlwe.plaintext, !lattigo.ckks.linear_transformation, !lattigo.ckks.linear_transformation>

  // CHECK: %[[c1_0:.*]] = arith.constant 1 : index
  // CHECK: %[[res:.*]] = memref.load %[[storage_lt]][%[[c1_0]]] : memref<2x![[LT]]>
  %res = preprocessing.load %storage[] site 2 <!lattigo.ckks.linear_transformation> : !preprocessing.storage<!lattigo.rlwe.plaintext, !lattigo.ckks.linear_transformation, !lattigo.ckks.linear_transformation>, !lattigo.ckks.linear_transformation
  return %res : !lattigo.ckks.linear_transformation
}
