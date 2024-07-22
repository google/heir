// RUN: heir-opt --secretize=entry-function=main --wrap-generic \
// RUN:   --align-tensor-sizes --canonicalize --cse --split-input-file %s | FileCheck %s

module  {
  // CHECK-LABEL: @main
  // CHECK-SAME{LITERAL}: !secret.secret<tensor<1024xi16, #tensor_ext.simd_packing<in = [16], padding = [0], out = [1024]>>>
  func.func @main(%arg0: tensor<16xi16>, %arg1: tensor<16xi16>) -> tensor<16xi16> {
    // CHECK-NEXT: secret.generic ins(%[[arg0:.*]], %[[arg1:.*]] : !secret.secret<[[ty:.*]]>, !secret.secret<[[ty]]>)
    // CHECK-NEXT:  ^bb0(%[[arg0_0:.*]]: [[ty]], %[[arg1_0:.*]]: [[ty]]):
    // CHECK-NEXT:    %[[add:.*]] = arith.addi %[[arg0_0]], %[[arg1_0]] : [[ty]]
    // CHECK-NEXT:    secret.yield %[[add]] : [[ty]]
    %0 = arith.addi %arg0, %arg1 : tensor<16xi16>
    return %0 : tensor<16xi16>
  }
}

// -----

module  {
  // CHECK-LABEL: @main
  // CHECK-SAME: %[[arg0:.*]]:
  // CHECK-SAME{LITERAL}: tensor<2x1024xi16, #tensor_ext.simd_packing<in = [2048], padding = [0], out = [2, 1024]>>
  func.func @main(%arg0: tensor<2048xi16>) -> tensor<2048xi16> {
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[c976:.*]] = arith.constant 976 : index
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c20:.*]] = arith.constant 20 : index
    // CHECK: secret.generic ins(%[[arg0]] : !secret.secret<[[ty:.*]]>)
    // CHECK-NEXT:  ^bb0(%[[arg0_0:.*]]: [[ty]]):
    // CHECK-NEXT:    %[[extract1:.*]] = tensor.extract %[[arg0_0]][%[[c1]], %[[c976]]] : [[ty]]
    // CHECK-NEXT:    %[[extracted:.*]] = tensor.insert %[[extract1]] into %[[arg0_0]][%[[c0]], %[[c20]]] : [[ty]]
    // CHECK-NEXT:    secret.yield %[[extracted]]
    %c20 = arith.constant 20 : index
    %c2000 = arith.constant 2000 : index
    %1 = tensor.extract %arg0[%c2000] : tensor<2048xi16>
    %inserted = tensor.insert %1 into %arg0[%c20] : tensor<2048xi16>
    return %inserted : tensor<2048xi16>
  }
}
