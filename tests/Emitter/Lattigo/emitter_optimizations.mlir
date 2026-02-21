// RUN: heir-translate %s --emit-lattigo --split-input-file | FileCheck %s

module attributes {scheme.bgv} {
  // CHECK: func test_splat_zero
  func.func @test_splat_zero() -> tensor<100xi32> {
    // CHECK: [[v0:[^ ]*]] := make([]int32, 100)
    // CHECK: return [[v0]]
    %c0 = arith.constant dense<0> : tensor<100xi32>
    return %c0 : tensor<100xi32>
  }
}

// -----

module attributes {scheme.bgv} {
  // CHECK: func test_splat_ones
  func.func @test_splat_ones() -> tensor<100xi32> {
    // CHECK: [[v0:[^ ]*]] := slices.Repeat([]int32{1}, 100)
    // CHECK: return [[v0]]
    %c1 = arith.constant dense<1> : tensor<100xi32>
    return %c1 : tensor<100xi32>
  }
}

// -----

module attributes {scheme.bgv} {
  // CHECK: func test_contiguous_extract_1d
  func.func @test_contiguous_extract_1d(%arg0: tensor<100xi32>) -> tensor<10xi32> {
    // CHECK: [[v0:[^ ]*]] := [[arg0:[^ ]*]][10 : 10 + 10]
    // CHECK: return [[v0]]
    %v = tensor.extract_slice %arg0[10] [10] [1] : tensor<100xi32> to tensor<10xi32>
    return %v : tensor<10xi32>
  }
}

// -----

module attributes {scheme.bgv} {
  // CHECK: func test_contiguous_extract_2d
  func.func @test_contiguous_extract_2d(%arg0: tensor<16x16xi32>) -> tensor<1x16xi32> {
    // CHECK: [[v0:[^ ]*]] := [[arg0:[^ ]*]][0 + 16 * (5) : 0 + 16 * (5) + 16]
    // CHECK: return [[v0]]
    %v = tensor.extract_slice %arg0[5, 0] [1, 16] [1, 1] : tensor<16x16xi32> to tensor<1x16xi32>
    return %v : tensor<1x16xi32>
  }
}

// -----

module attributes {scheme.bgv} {
  // CHECK: func test_contiguous_insert_1d
  func.func @test_contiguous_insert_1d(%arg0: tensor<100xi32>, %arg1: tensor<10xi32>) -> tensor<100xi32> {
    // CHECK: [[v_insert:[^ ]*]] := append(make([]int32, 0, len([[arg0:[^ ]*]])), [[arg0]]...)
    // CHECK: copy([[v_insert]][20:], [[arg1:[^ ]*]])
    // CHECK: return [[v_insert]]
    %v = tensor.insert_slice %arg1 into %arg0[20] [10] [1] : tensor<10xi32> into tensor<100xi32>
    return %v : tensor<100xi32>
  }
}

// -----

module attributes {scheme.bgv} {
  // CHECK: func test_contiguous_insert_2d
  func.func @test_contiguous_insert_2d(%arg0: tensor<16x16xi32>, %arg1: tensor<1x16xi32>) -> tensor<16x16xi32> {
    // CHECK: [[v_insert:[^ ]*]] := append(make([]int32, 0, len([[arg0:[^ ]*]])), [[arg0]]...)
    // CHECK: copy([[v_insert]][0 + 16 * (8):], [[arg1:[^ ]*]])
    // CHECK: return [[v_insert]]
    %v = tensor.insert_slice %arg1 into %arg0[8, 0] [1, 16] [1, 1] : tensor<1x16xi32> into tensor<16x16xi32>
    return %v : tensor<16x16xi32>
  }
}
