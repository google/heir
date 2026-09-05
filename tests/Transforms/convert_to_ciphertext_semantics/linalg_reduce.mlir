// RUN: heir-opt --layout-propagation=min-slot-count=8192 --convert-to-ciphertext-semantics=min-slot-count=8192 %s | FileCheck %s

#bicyclic = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (65i0 - 66i1 + slot) mod 2145 = 0 and 0 <= i0 <= 32 and 0 <= i1 <= 64 and 0 <= slot <= 8191 }">

module {
  // Test that an 8-length vector gets reduced.
  // CHECK: func.func @main
  // CHECK-NOT: linalg.reduce
  // CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[ASSIGN:.*]] = arith.constant dense<0{{.*}}> : tensor<1x8192xf32>
  // CHECK: tensor_ext.rotate %{{.*}}, %[[c1]]
  // CHECK: tensor_ext.rotate %{{.*}}, %[[c2]]
  // CHECK: tensor_ext.rotate %{{.*}}, %[[c4]]
  // CHECK: arith.addf %{{.*}}, %[[ASSIGN]]
  func.func @main(%arg0: !secret.secret<tensor<8xf32>>, %arg1: !secret.secret<tensor<8xf32>>) -> !secret.secret<tensor<f32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<8xf32>>, %arg1: !secret.secret<tensor<8xf32>>) {
    ^body(%input0: tensor<8xf32>, %input1: tensor<8xf32>):
      %1 = arith.mulf %input0, %input1 : tensor<8xf32>
      %reduced = linalg.reduce ins(%1 : tensor<8xf32>) outs(%cst : tensor<f32>) dimensions = [0]
        (%in: f32, %init: f32) {
          %2 = arith.addf %in, %init : f32
          linalg.yield %2 : f32
        }
      secret.yield %reduced : tensor<f32>
    } -> !secret.secret<tensor<f32>>
    return %0 : !secret.secret<tensor<f32>>
  }

  // CHECK: func.func @reduce_bicyclic
  // CHECK-DAG: %[[C33:.*]] = arith.constant 33 : index
  // CHECK-DAG: %[[C66:.*]] = arith.constant 66 : index
  // CHECK-DAG: %[[C132:.*]] = arith.constant 132 : index
  // CHECK-DAG: %[[C264:.*]] = arith.constant 264 : index
  // CHECK-DAG: %[[C528:.*]] = arith.constant 528 : index
  // CHECK-DAG: %[[C1056:.*]] = arith.constant 1056 : index
  // CHECK: tensor_ext.rotate {{.*}}, %[[C33]]
  // CHECK: tensor_ext.rotate {{.*}}, %[[C66]]
  // CHECK: tensor_ext.rotate {{.*}}, %[[C132]]
  // CHECK: tensor_ext.rotate {{.*}}, %[[C264]]
  // CHECK: tensor_ext.rotate {{.*}}, %[[C528]]
  // CHECK: tensor_ext.rotate {{.*}}, %[[C1056]]
  // CHECK: tensor_ext.remap
  func.func @reduce_bicyclic(%arg0: !secret.secret<tensor<33x65xf32>> {tensor_ext.layout = #bicyclic}) -> !secret.secret<tensor<33xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<33xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<33x65xf32>>) {
    ^body(%input0: tensor<33x65xf32>):
      %reduced = linalg.reduce ins(%input0 : tensor<33x65xf32>) outs(%cst : tensor<33xf32>) dimensions = [1]
        (%in: f32, %init: f32) {
          %1 = arith.addf %in, %init : f32
          linalg.yield %1 : f32
        }
      secret.yield %reduced : tensor<33xf32>
    } -> !secret.secret<tensor<33xf32>>
    return %0 : !secret.secret<tensor<33xf32>>
  }
}
