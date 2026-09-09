// RUN: heir-opt --layout-propagation=min-slot-count=8192 %s | FileCheck %s

// CHECK-DAG: #[[reduced_layout:.*]] = #tensor_ext.layout<"{ [] -> [ct, slot] : ct = 0 and 0 <= slot <= 8191 }">
// CHECK-DAG: #[[input_layout:.*]] = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 8 = 0 and 0 <= i0 <= 7 and 0 <= slot <= 8191 }">
// CHECK-DAG: #[[REDUCED_LAYOUT:.*]] = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 33 = 0 and 0 <= i0 <= 32 and 0 <= slot <= 8191 }">

#bicyclic = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (65i0 - 66i1 + slot) mod 2145 = 0 and 0 <= i0 <= 32 and 0 <= i1 <= 64 and 0 <= slot <= 8191 }">

module {
  // CHECK: func.func @main
  // CHECK-SAME: %{{.*}}: !secret.secret<tensor<8xf32>> {{{.*}}tensor_ext.layout = #[[input_layout]]},
  // CHECK-SAME: %{{.*}}: !secret.secret<tensor<8xf32>> {{{.*}}tensor_ext.layout = #[[input_layout]]}
  // CHECK-SAME: -> (!secret.secret<tensor<f32>> {tensor_ext.layout = #[[reduced_layout]]})
  func.func @main(%arg0: !secret.secret<tensor<8xf32>>, %arg1: !secret.secret<tensor<8xf32>>) -> !secret.secret<tensor<f32>> {
    // CHECK-DAG: %[[cst:.*]] = arith.constant
    // CHECK-DAG: tensor_ext.assign_layout %[[cst]]
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
  // CHECK: ^body(%[[INPUT:.*]]: tensor<33x65xf32>):
  // CHECK: %[[INIT:.*]] = tensor_ext.assign_layout
  // CHECK: linalg.reduce ins(%[[INPUT]] : tensor<33x65xf32>) outs(%[[INIT]] : tensor<33xf32>)
  // CHECK-NOT: tensor_ext.convert_layout
  // CHECK-SAME: tensor_ext.layout = #[[REDUCED_LAYOUT]]
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
