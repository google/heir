// RUN: heir-opt %s --layout-propagation=min-slot-count=32 --convert-to-ciphertext-semantics=min-slot-count=32 | FileCheck %s

// Summing the rows of a row-major 4x8 matrix (dimension 0) steps through
// the matrix 8 slots at a time: rotations by 8 and 16.

// CHECK: func.func @column_sum
// CHECK-DAG: %[[c8:.*]] = arith.constant 8 : index
// CHECK-DAG: %[[c16:.*]] = arith.constant 16 : index
// CHECK: tensor_ext.rotate %{{.*}}, %[[c8]]
// CHECK: tensor_ext.rotate %{{.*}}, %[[c16]]
// CHECK-NOT: tensor_ext.rotate
// CHECK: return
func.func @column_sum(%x: !secret.secret<tensor<4x8xf32>>) -> !secret.secret<tensor<8xf32>> {
  %zero = arith.constant dense<0.0> : tensor<8xf32>
  %0 = secret.generic(%x: !secret.secret<tensor<4x8xf32>>) {
  ^body(%input: tensor<4x8xf32>):
    %sum = linalg.reduce { arith.addf } ins(%input : tensor<4x8xf32>) outs(%zero : tensor<8xf32>) dimensions = [0]
    secret.yield %sum : tensor<8xf32>
  } -> !secret.secret<tensor<8xf32>>
  return %0 : !secret.secret<tensor<8xf32>>
}
