// RUN: heir-opt --rotation-analysis --split-input-file %s | FileCheck %s

// CHECK: module attributes
// here 14 == -2 mod 16
// CHECK-SAME: rotation_analysis.indices = array<i64: 0, 2, 4, 6, 8, 14>
module {
  func.func @test_loop(%arg0: tensor<16xi32>) -> tensor<16xi32> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cm1 = arith.constant -1 : index
    %c5 = arith.constant 5 : index
    %0 = scf.for %i = %cm1 to %c5 step %c1 iter_args(%iter = %arg0) -> (tensor<16xi32>) {
       %rot = arith.muli %c2, %i : index
       %2 = tensor_ext.rotate %arg0, %rot : tensor<16xi32>, index
       scf.yield %2 : tensor<16xi32>
    }
    return %0 : tensor<16xi32>
  }
}

// -----

// CHECK: module attributes
// CHECK-SAME: rotation_analysis.indices = array<i64: 0, 2, 4, 6, 8, 10, 12>
module {
  func.func @test_nested_loop(%arg0: tensor<16xi32>) -> tensor<16xi32> {
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    // i = 0, 2, 4
    %0 = scf.for %i = %c0 to %c5 step %c2 iter_args(%iter = %arg0) -> (tensor<16xi32>) {
        // j = 0, 2, 4
        %1 = scf.for %j = %c0 to %c5 step %c2 iter_args(%iter2 = %iter) -> (tensor<16xi32>) {
           // shift = 2*i + j
           %z = arith.muli %c2, %i : index
           %shift = arith.addi %z, %j : index
           %2 = tensor_ext.rotate %iter2, %shift : tensor<16xi32>, index
           scf.yield %2 : tensor<16xi32>
        }
       scf.yield %1 : tensor<16xi32>
    }
    return %0 : tensor<16xi32>
  }
}
