// RUN: heir-opt --preprocessing-to-openfhe %s | FileCheck %s

// CHECK: ![[PT:.*]] = !openfhe.plaintext

// CHECK: func @test_openfhe
// CHECK-SAME: (%[[arg0:.*]]: ![[PT]]) -> ![[PT]]
func.func @test_openfhe(%arg0: !openfhe.plaintext) -> !openfhe.plaintext {
  // CHECK: %[[storage:.*]] = memref.alloc() : memref<2x![[PT]]>
  %storage = preprocessing.empty : !preprocessing.storage<!openfhe.plaintext, !openfhe.plaintext>

  // CHECK: %[[c0:.*]] = arith.constant 0 : index
  // CHECK: memref.store %[[arg0]], %[[storage]][%[[c0]]] : memref<2x![[PT]]>
  preprocessing.store %arg0, %storage[] site 0 <!openfhe.plaintext> : !openfhe.plaintext, !preprocessing.storage<!openfhe.plaintext, !openfhe.plaintext>

  // CHECK: %[[c1:.*]] = arith.constant 1 : index
  // CHECK: memref.store %[[arg0]], %[[storage]][%[[c1]]] : memref<2x![[PT]]>
  preprocessing.store %arg0, %storage[] site 1 <!openfhe.plaintext> : !openfhe.plaintext, !preprocessing.storage<!openfhe.plaintext, !openfhe.plaintext>

  // CHECK: %[[c0_1:.*]] = arith.constant 0 : index
  // CHECK: %[[res:.*]] = memref.load %[[storage]][%[[c0_1]]] : memref<2x![[PT]]>
  %res = preprocessing.load %storage[] site 0 <!openfhe.plaintext> : !preprocessing.storage<!openfhe.plaintext, !openfhe.plaintext>, !openfhe.plaintext
  return %res : !openfhe.plaintext
}
