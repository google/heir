// RUN: heir-translate %s --emit-lattigo | FileCheck %s

module attributes {scheme.ckks} {
  // CHECK: func test_dynamic_extract_slice
  // CHECK: ([[arg0:v[0-9]+]] []float32, [[arg1:v[0-9]+]] int64)
  // CHECK: [[v2:v[0-9]+]] := [[arg0]][0 : 0 + [[arg1]]]
  // CHECK: return [[v2]]
  func.func @test_dynamic_extract_slice(%arg0: tensor<1024xf32>, %arg1: index) -> tensor<?xf32> {
    %0 = tensor.extract_slice %arg0[0] [%arg1] [1] : tensor<1024xf32> to tensor<?xf32>
    return %0 : tensor<?xf32>
  }

  // CHECK: func test_dynamic_insert_slice
  // CHECK: ([[arg0:v[0-9]+]] []float32, [[arg1:v[0-9]+]] []float32, [[arg2:v[0-9]+]] int64, [[arg3:v[0-9]+]] int64)
  // CHECK: [[dest:v[0-9]+]] := append(make([]float32, 0, len([[arg1]])), [[arg1]]...)
  // CHECK: copy([[dest]][[[arg2]] + 1024 * (0) : [[arg2]] + 1024 * (0) + 1 * [[arg3]]], [[arg0]])
  // CHECK: return [[dest]]
  func.func @test_dynamic_insert_slice(%arg0: tensor<1x?xf32>, %arg1: tensor<1x1024xf32>, %arg2: index, %arg3: index) -> tensor<1x1024xf32> {
    %0 = tensor.insert_slice %arg0 into %arg1[0, %arg2] [1, %arg3] [1, 1] : tensor<1x?xf32> into tensor<1x1024xf32>
    return %0 : tensor<1x1024xf32>
  }

  // Test the case from the user request
  // CHECK: func test_user_request
  func.func @test_user_request(%arg0: tensor<512x1024xf32>, %arg2: index) -> tensor<1024xf32> {
    %c1024 = arith.constant 1024 : index
    %c-23 = arith.constant -23 : index
    %8 = arith.muli %arg2, %c-23 : index
    %9 = arith.remsi %8, %c1024 : index
    %10 = arith.addi %9, %c1024 : index
    %11 = arith.remsi %10, %c1024 : index
    %12 = arith.subi %c1024, %11 : index
    // CHECK: [[v9:v[0-9]+]] := [[v0:v[0-9]+]][0 + 1024 * (0) : 0 + 1024 * (0) + 1 * [[v7:v[0-9]+]]]
    %extracted_slice_21 = tensor.extract_slice %arg0[0, 0] [1, %11] [1, 1] : tensor<512x1024xf32> to tensor<1x?xf32>
    // CHECK: [[v10:v[0-9]+]] := [[v0]][[[v7]] + 1024 * (0) : [[v7]] + 1024 * (0) + 1 * [[v8:v[0-9]+]]]
    %extracted_slice_22 = tensor.extract_slice %arg0[0, %11] [1, %12] [1, 1] : tensor<512x1024xf32> to tensor<1x?xf32>
    %13 = tensor.empty() : tensor<1x1024xf32>
    // CHECK: [[v12_dest:v[0-9]+]] := append(make([]float32, 0, len([[v11_base:v[0-9]+]])), [[v11_base]]...)
    // CHECK: copy([[v12_dest]][[[v8]] + 1024 * (0) : [[v8]] + 1024 * (0) + 1 * [[v7]]], [[v9]])
    %inserted_slice = tensor.insert_slice %extracted_slice_21 into %13[0, %12] [1, %11] [1, 1] : tensor<1x?xf32> into tensor<1x1024xf32>
    // CHECK: [[v13_dest:v[0-9]+]] := append(make([]float32, 0, len([[v12_dest]])), [[v12_dest]]...)
    // CHECK: copy([[v13_dest]][0 + 1024 * (0) : 0 + 1024 * (0) + 1 * [[v8]]], [[v10]])
    %inserted_slice_23 = tensor.insert_slice %extracted_slice_22 into %inserted_slice[0, 0] [1, %12] [1, 1] : tensor<1x?xf32> into tensor<1x1024xf32>
    %extracted_slice_24 = tensor.extract_slice %inserted_slice_23[0, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> to tensor<1024xf32>
    return %extracted_slice_24 : tensor<1024xf32>
  }
}
