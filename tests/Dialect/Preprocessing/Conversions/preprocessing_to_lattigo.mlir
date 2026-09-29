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

// CHECK: ![[LT:.*]] = !lattigo.ckks.linear_transformation

// CHECK: func @test_lattigo_prepared_linear_transform_only
// CHECK-SAME: (%[[lt:.*]]: ![[LT]]) -> ![[LT]]
func.func @test_lattigo_prepared_linear_transform_only(
    %lt: !lattigo.ckks.linear_transformation) -> !lattigo.ckks.linear_transformation {
  // CHECK: %[[storage_lt:.*]] = memref.alloc() : memref<2x![[LT]]>
  %storage = preprocessing.empty : !preprocessing.storage<!lattigo.ckks.linear_transformation, !lattigo.ckks.linear_transformation>

  // CHECK: %[[c0:.*]] = arith.constant 0 : index
  // CHECK: memref.store %[[lt]], %[[storage_lt]][%[[c0]]] : memref<2x![[LT]]>
  preprocessing.store %lt, %storage[] site 0 <!lattigo.ckks.linear_transformation> : !lattigo.ckks.linear_transformation, !preprocessing.storage<!lattigo.ckks.linear_transformation, !lattigo.ckks.linear_transformation>

  // CHECK: %[[c1:.*]] = arith.constant 1 : index
  // CHECK: memref.store %[[lt]], %[[storage_lt]][%[[c1]]] : memref<2x![[LT]]>
  preprocessing.store %lt, %storage[] site 1 <!lattigo.ckks.linear_transformation> : !lattigo.ckks.linear_transformation, !preprocessing.storage<!lattigo.ckks.linear_transformation, !lattigo.ckks.linear_transformation>

  // CHECK: %[[c1_0:.*]] = arith.constant 1 : index
  // CHECK: %[[res:.*]] = memref.load %[[storage_lt]][%[[c1_0]]] : memref<2x![[LT]]>
  %res = preprocessing.load %storage[] site 1 <!lattigo.ckks.linear_transformation> : !preprocessing.storage<!lattigo.ckks.linear_transformation, !lattigo.ckks.linear_transformation>, !lattigo.ckks.linear_transformation
  return %res : !lattigo.ckks.linear_transformation
}

// -----

// CHECK-DAG: ![[LT:.*]] = !lattigo.ckks.linear_transformation
// CHECK-DAG: ![[PT2:.*]] = !lattigo.rlwe.plaintext

// CHECK: func @test_lattigo_multi_type
// CHECK-SAME: (%[[pt:.*]]: ![[PT2]], %[[lt:.*]]: ![[LT]]) -> (![[LT]], ![[LT]])
func.func @test_lattigo_multi_type(
    %pt: !lattigo.rlwe.plaintext,
    %lt: !lattigo.ckks.linear_transformation) -> (!lattigo.ckks.linear_transformation, !lattigo.ckks.linear_transformation) {
  // CHECK-DAG: %[[storage_pt:.*]] = memref.alloc() : memref<1x![[PT2]]>
  // CHECK-DAG: %[[storage_lt:.*]] = memref.alloc() : memref<2x![[LT]]>
  %storage = preprocessing.empty : !preprocessing.storage<!lattigo.rlwe.plaintext, !lattigo.ckks.linear_transformation, !lattigo.ckks.linear_transformation>

  // CHECK: %[[c0:.*]] = arith.constant 0 : index
  // CHECK: memref.store %[[pt]], %[[storage_pt]][%[[c0]]] : memref<1x![[PT2]]>
  preprocessing.store %pt, %storage[] site 0 <!lattigo.rlwe.plaintext> : !lattigo.rlwe.plaintext, !preprocessing.storage<!lattigo.rlwe.plaintext, !lattigo.ckks.linear_transformation, !lattigo.ckks.linear_transformation>

  // CHECK: %[[c0_0:.*]] = arith.constant 0 : index
  // CHECK: memref.store %[[lt]], %[[storage_lt]][%[[c0_0]]] : memref<2x![[LT]]>
  preprocessing.store %lt, %storage[] site 1 <!lattigo.ckks.linear_transformation> : !lattigo.ckks.linear_transformation, !preprocessing.storage<!lattigo.rlwe.plaintext, !lattigo.ckks.linear_transformation, !lattigo.ckks.linear_transformation>

  // CHECK: %[[c1:.*]] = arith.constant 1 : index
  // CHECK: memref.store %[[lt]], %[[storage_lt]][%[[c1]]] : memref<2x![[LT]]>
  preprocessing.store %lt, %storage[] site 2 <!lattigo.ckks.linear_transformation> : !lattigo.ckks.linear_transformation, !preprocessing.storage<!lattigo.rlwe.plaintext, !lattigo.ckks.linear_transformation, !lattigo.ckks.linear_transformation>

  // CHECK: %[[c0_1:.*]] = arith.constant 0 : index
  // CHECK: %[[res1:.*]] = memref.load %[[storage_lt]][%[[c0_1]]] : memref<2x![[LT]]>
  %res1 = preprocessing.load %storage[] site 1 <!lattigo.ckks.linear_transformation> : !preprocessing.storage<!lattigo.rlwe.plaintext, !lattigo.ckks.linear_transformation, !lattigo.ckks.linear_transformation>, !lattigo.ckks.linear_transformation

  // CHECK: %[[c1_0:.*]] = arith.constant 1 : index
  // CHECK: %[[res2:.*]] = memref.load %[[storage_lt]][%[[c1_0]]] : memref<2x![[LT]]>
  %res2 = preprocessing.load %storage[] site 2 <!lattigo.ckks.linear_transformation> : !preprocessing.storage<!lattigo.rlwe.plaintext, !lattigo.ckks.linear_transformation, !lattigo.ckks.linear_transformation>, !lattigo.ckks.linear_transformation
  return %res1, %res2 : !lattigo.ckks.linear_transformation, !lattigo.ckks.linear_transformation
}

// -----

// CHECK-DAG: ![[LT:.*]] = !lattigo.ckks.linear_transformation
// CHECK-DAG: ![[PT:.*]] = !lattigo.rlwe.plaintext

// CHECK: func @callee_storage
// CHECK-SAME: (%[[pt:.*]]: ![[PT]], %[[lt:.*]]: ![[LT]]) -> (memref<1x![[PT]]>, memref<1x![[LT]]>)
func.func @callee_storage(%pt: !lattigo.rlwe.plaintext, %lt: !lattigo.ckks.linear_transformation)
    -> !preprocessing.storage<!lattigo.rlwe.plaintext, !lattigo.ckks.linear_transformation> {
  // CHECK-DAG: %[[storage_pt:.*]] = memref.alloc() : memref<1x![[PT]]>
  // CHECK-DAG: %[[storage_lt:.*]] = memref.alloc() : memref<1x![[LT]]>
  %storage = preprocessing.empty : !preprocessing.storage<!lattigo.rlwe.plaintext, !lattigo.ckks.linear_transformation>
  // CHECK: %[[c0:.*]] = arith.constant 0 : index
  // CHECK: memref.store %[[pt]], %[[storage_pt]][%[[c0]]] : memref<1x![[PT]]>
  preprocessing.store %pt, %storage[] site 0 <!lattigo.rlwe.plaintext> : !lattigo.rlwe.plaintext, !preprocessing.storage<!lattigo.rlwe.plaintext, !lattigo.ckks.linear_transformation>
  // CHECK: %[[c0_0:.*]] = arith.constant 0 : index
  // CHECK: memref.store %[[lt]], %[[storage_lt]][%[[c0_0]]] : memref<1x![[LT]]>
  preprocessing.store %lt, %storage[] site 1 <!lattigo.ckks.linear_transformation> : !lattigo.ckks.linear_transformation, !preprocessing.storage<!lattigo.rlwe.plaintext, !lattigo.ckks.linear_transformation>
  // CHECK: return %[[storage_pt]], %[[storage_lt]] : memref<1x![[PT]]>, memref<1x![[LT]]>
  return %storage : !preprocessing.storage<!lattigo.rlwe.plaintext, !lattigo.ckks.linear_transformation>
}

// CHECK: func @caller_storage
// CHECK-SAME: (%[[pt:.*]]: ![[PT]], %[[lt:.*]]: ![[LT]]) -> (![[PT]], ![[LT]])
func.func @caller_storage(%pt: !lattigo.rlwe.plaintext, %lt: !lattigo.ckks.linear_transformation)
    -> (!lattigo.rlwe.plaintext, !lattigo.ckks.linear_transformation) {
  // CHECK: %[[res_storage:.*]]:2 = call @callee_storage(%[[pt]], %[[lt]]) : (![[PT]], ![[LT]]) -> (memref<1x![[PT]]>, memref<1x![[LT]]>)
  %storage = func.call @callee_storage(%pt, %lt) : (!lattigo.rlwe.plaintext, !lattigo.ckks.linear_transformation)
      -> !preprocessing.storage<!lattigo.rlwe.plaintext, !lattigo.ckks.linear_transformation>
  // CHECK: %[[c0:.*]] = arith.constant 0 : index
  // CHECK: %[[loaded_pt:.*]] = memref.load %[[res_storage]]#0[%[[c0]]] : memref<1x![[PT]]>
  %res_pt = preprocessing.load %storage[] site 0 <!lattigo.rlwe.plaintext> : !preprocessing.storage<!lattigo.rlwe.plaintext, !lattigo.ckks.linear_transformation>, !lattigo.rlwe.plaintext
  // CHECK: %[[c0_0:.*]] = arith.constant 0 : index
  // CHECK: %[[loaded_lt:.*]] = memref.load %[[res_storage]]#1[%[[c0_0]]] : memref<1x![[LT]]>
  %res_lt = preprocessing.load %storage[] site 1 <!lattigo.ckks.linear_transformation> : !preprocessing.storage<!lattigo.rlwe.plaintext, !lattigo.ckks.linear_transformation>, !lattigo.ckks.linear_transformation
  // CHECK: return %[[loaded_pt]], %[[loaded_lt]] : ![[PT]], ![[LT]]
  return %res_pt, %res_lt : !lattigo.rlwe.plaintext, !lattigo.ckks.linear_transformation
}
