// RUN: heir-opt --forward-insert-slice-to-extract-slice %s | FileCheck %s

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK: func @forward_disjoint_slices
// CHECK-SAME: (%[[arg:.*]]: tensor<2x2xf32>)
func.func @forward_disjoint_slices(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %initial_tensor = tensor.empty() : tensor<4x4xf32>

  // Insert a 2x2 slice at (0, 0)
  %inserted_tensor = tensor.insert_slice %arg0 into %initial_tensor[0, 0][2, 2][1, 1] : tensor<2x2xf32> into tensor<4x4xf32>

  // Extract a 2x2 slice at (2, 2) which is disjoint from the inserted slice
  %extracted_disjoint = tensor.extract_slice %inserted_tensor[2, 2][2, 2][1, 1] : tensor<4x4xf32> to tensor<2x2xf32>

  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<2x2xf32>
  // CHECK: return %[[EMPTY]] : tensor<2x2xf32>
  return %extracted_disjoint : tensor<2x2xf32>
}

// CHECK: func @forward_overlapping_slices
// CHECK-SAME: (%[[arg:.*]]: tensor<2x2xf32>)
func.func @forward_overlapping_slices(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %initial_tensor = tensor.empty() : tensor<4x4xf32>

  // Insert a 2x2 slice at (0, 0)
  %inserted_tensor = tensor.insert_slice %arg0 into %initial_tensor[0, 0][2, 2][1, 1] : tensor<2x2xf32> into tensor<4x4xf32>

  // Extract a 2x2 slice at (0, 0) which is the same as the inserted slice
  %extracted_overlapping = tensor.extract_slice %inserted_tensor[0, 0][2, 2][1, 1] : tensor<4x4xf32> to tensor<2x2xf32>

  // CHECK: return %[[arg]] : tensor<2x2xf32>
  return %extracted_overlapping : tensor<2x2xf32>
}

// CHECK: func @forward_partially_overlapping_slices
func.func @forward_partially_overlapping_slices() -> tensor<2x2xf32> {
  %initial_tensor = tensor.empty() : tensor<4x4xf32>
  %source_tensor = tensor.empty() : tensor<2x2xf32>

  // Insert a 2x2 slice at (0, 0)
  %inserted_tensor = tensor.insert_slice %source_tensor into %initial_tensor[0, 0][2, 2][1, 1] : tensor<2x2xf32> into tensor<4x4xf32>

  // Extract a 2x2 slice at (1, 1) which partially overlaps the inserted slice
  %extracted_partial = tensor.extract_slice %inserted_tensor[1, 1][2, 2][1, 1] : tensor<4x4xf32> to tensor<2x2xf32>

  // CHECK: %[[INSERTED:.*]] = tensor.insert_slice %[[SOURCE_TENSOR:.*]] into %[[INITIAL_TENSOR:.*]][0, 0] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<4x4xf32>
  // CHECK: %[[EXTRACTED:.*]] = tensor.extract_slice %[[INSERTED]][1, 1] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
  // CHECK: return %[[EXTRACTED]] : tensor<2x2xf32>
  return %extracted_partial : tensor<2x2xf32>
}
