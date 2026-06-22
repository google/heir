// RUN: heir-opt --preprocessing-to-lattigo %s | FileCheck %s

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
