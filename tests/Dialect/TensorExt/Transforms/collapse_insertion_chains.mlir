// RUN: heir-opt --collapse-insertion-chains --canonicalize %s | FileCheck %s

// CHECK: @test_collapse_insertion_chains
// CHECK-SAME: (%[[in:.*]]: tensor<4xi32>, %[[out:.*]]: tensor<4xi32>) -> tensor<4xi32> {
// CHECK: %[[c2:.*]] = arith.constant 2 : index
// CHECK: %[[res:.*]] = tensor_ext.rotate %[[in]], %[[c2]] : tensor<4xi32>, index
// CHECK: return %[[res]] : tensor<4xi32>
func.func @test_collapse_insertion_chains(%in: tensor<4xi32>, %out: tensor<4xi32>) -> tensor<4xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %ex0 = tensor.extract %in[%c3] : tensor<4xi32>
  %in0 = tensor.insert %ex0 into %out[%c1] : tensor<4xi32>
  %ex1 = tensor.extract %in[%c0] : tensor<4xi32>
  %in1 = tensor.insert %ex1 into %in0[%c2] : tensor<4xi32>
  %ex2 = tensor.extract %in[%c1] : tensor<4xi32>
  %in2 = tensor.insert %ex2 into %in1[%c3] : tensor<4xi32>
  %ex3 = tensor.extract %in[%c2] : tensor<4xi32>
  %in3 = tensor.insert %ex3 into %in2[%c0] : tensor<4xi32>
  return %in3 : tensor<4xi32>
}
