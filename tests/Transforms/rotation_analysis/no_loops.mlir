// RUN: heir-opt --rotation-analysis --split-input-file %s | FileCheck %s

// CHECK: module attributes
// CHECK-SAME: rotation_analysis.indices = array<i64: 4>
module {
  func.func @test_one_shift(%arg0: tensor<16xi32>) -> tensor<16xi32> {
    %shift = arith.constant 4 : i32
    %2 = tensor_ext.rotate %arg0, %shift : tensor<16xi32>, i32
    return %2 : tensor<16xi32>
  }
}

// -----

// CHECK: module attributes
// CHECK-SAME: rotation_analysis.indices = array<i64: 4, 5>
module {
  func.func @test_two_shifts(%arg0: tensor<16xi32>) -> tensor<16xi32> {
    %shift = arith.constant 4 : i32
    %shift2 = arith.constant 5 : i32
    %2 = tensor_ext.rotate %arg0, %shift : tensor<16xi32>, i32
    %3 = tensor_ext.rotate %arg0, %shift2 : tensor<16xi32>, i32
    %4 = arith.addi %2, %3 : tensor<16xi32>
    return %4 : tensor<16xi32>
  }
}

// -----

// CHECK: module attributes
// CHECK-SAME: rotation_analysis.indices = array<i64: 9>
module {
  func.func @test_computed_shift(%arg0: tensor<16xi32>) -> tensor<16xi32> {
    %c4 = arith.constant 4 : i32
    %c5 = arith.constant 5 : i32
    %shift = arith.addi %c4, %c5 : i32
    %2 = tensor_ext.rotate %arg0, %shift : tensor<16xi32>, i32
    return %2 : tensor<16xi32>
  }
}

// -----

// CHECK: module attributes
// CHECK-SAME: rotation_analysis.indices = array<i64: 2>
module {
  func.func @test_div(%arg0: tensor<16xi32>) -> tensor<16xi32> {
    %c8 = arith.constant 8 : i32
    %c4 = arith.constant 4 : i32
    %shift = arith.divsi %c8, %c4 : i32
    %2 = tensor_ext.rotate %arg0, %shift : tensor<16xi32>, i32
    return %2 : tensor<16xi32>
  }
}

// -----

// CHECK: module attributes
// CHECK-SAME: rotation_analysis.indices = array<i64: 7>
module {
  func.func @test_extract_from_constant_tensor(%arg0: tensor<16xi32>) -> tensor<16xi32> {
    %shifts = arith.constant dense<[5, 3, 7, 8]> : tensor<4xindex>
    %idx = arith.constant 2 : index
    %shift = tensor.extract %shifts[%idx] : tensor<4xindex>
    %1 = tensor_ext.rotate %arg0, %shift : tensor<16xi32>, index
    return %1 : tensor<16xi32>
  }
}
