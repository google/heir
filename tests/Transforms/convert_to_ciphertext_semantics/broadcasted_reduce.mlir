// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics=min-slot-count=8 | FileCheck %s

// Test natural replication: N = B * period (8 = 8 * 1)
// B = 8 (dimension 0 size), period = 1, N = 8 (ciphertext size)
// CHECK: func.func @test_natural_replication
// CHECK-NOT: tensor_ext.broadcasted_reduce
// CHECK-NOT: arith.constant dense
// CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
// CHECK: %[[GENERIC:.*]] = secret.generic
// CHECK: %[[ROT4:.*]] = tensor_ext.rotate %{{.*}}, %[[c4]]
// CHECK: %[[ADD4:.*]] = arith.addf %{{.*}}, %[[ROT4]]
// CHECK: %[[ROT2:.*]] = tensor_ext.rotate %[[ADD4]], %[[c2]]
// CHECK: %[[ADD2:.*]] = arith.addf %[[ADD4]], %[[ROT2]]
// CHECK: %[[ROT1:.*]] = tensor_ext.rotate %[[ADD2]], %[[c1]]
// CHECK: %[[ADD1:.*]] = arith.addf %[[ADD2]], %[[ROT1]]
// CHECK: secret.yield %[[ADD1]]
// CHECK: return %[[GENERIC]]
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = i0 and 0 <= i0 <= 7 }">
module {
  func.func @test_natural_replication(%arg0: !secret.secret<tensor<8xf32>> {tensor_ext.layout = #layout}) -> (!secret.secret<tensor<8xf32>> {tensor_ext.layout = #layout}) {
    %0 = secret.generic(%arg0: !secret.secret<tensor<8xf32>> {tensor_ext.layout = #layout}) {
    ^body(%input: tensor<8xf32>):
      %reduced = tensor_ext.broadcasted_reduce %input {dimension = 0 : i64, reduceOp = "arith.addf", tensor_ext.layout = #layout} : tensor<8xf32>
      secret.yield %reduced : tensor<8xf32>
    } -> (!secret.secret<tensor<8xf32>> {tensor_ext.layout = #layout})
    return %0 : !secret.secret<tensor<8xf32>>
  }
}

// -----

// Test cleanup mask: N > B * period (8 > 4 * 1)
// B = 4 (dimension 0 size), period = 1, N = 8 (ciphertext size)
// CHECK: func.func @test_cleanup_mask
// CHECK-NOT: tensor_ext.broadcasted_reduce
// CHECK-DAG: %[[MASK:.*]] = arith.constant dense<{{\[\[}}0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00{{\]\]}}> : tensor<1x8xf32>
// CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[c5:.*]] = arith.constant 5 : index
// CHECK: %[[GENERIC:.*]] = secret.generic
// CHECK: %[[ROT2:.*]] = tensor_ext.rotate %{{.*}}, %[[c2]]
// CHECK: %[[ADD2:.*]] = arith.addf %{{.*}}, %[[ROT2]]
// CHECK: %[[ROT1:.*]] = tensor_ext.rotate %[[ADD2]], %[[c1]]
// CHECK: %[[ADD1:.*]] = arith.addf %[[ADD2]], %[[ROT1]]
// CHECK: %[[SHIFT:.*]] = tensor_ext.rotate %[[ADD1]], %[[c5]]
// CHECK: %[[CLEAN:.*]] = arith.mulf %[[SHIFT]], %[[MASK]]
// CHECK: %[[ROT1_REP:.*]] = tensor_ext.rotate %[[CLEAN]], %[[c1]]
// CHECK: %[[ADD1_REP:.*]] = arith.addf %[[CLEAN]], %[[ROT1_REP]]
// CHECK: %[[ROT2_REP:.*]] = tensor_ext.rotate %[[ADD1_REP]], %[[c2]]
// CHECK: %[[ADD2_REP:.*]] = arith.addf %[[ADD1_REP]], %[[ROT2_REP]]
// CHECK: secret.yield %[[ADD2_REP]]
// CHECK: return %[[GENERIC]]
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = i0 and 0 <= i0 <= 3 }">
module {
  func.func @test_cleanup_mask(%arg0: !secret.secret<tensor<4xf32>> {tensor_ext.layout = #layout}) -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #layout}) {
    %0 = secret.generic(%arg0: !secret.secret<tensor<4xf32>> {tensor_ext.layout = #layout}) {
    ^body(%input: tensor<4xf32>):
      %reduced = tensor_ext.broadcasted_reduce %input {dimension = 0 : i64, reduceOp = "arith.addf", tensor_ext.layout = #layout} : tensor<4xf32>
      secret.yield %reduced : tensor<4xf32>
    } -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #layout})
    return %0 : !secret.secret<tensor<4xf32>>
  }
}
