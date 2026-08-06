// RUN: heir-opt --split-input-file --verify-diagnostics %s

// -----

func.func @test_diagonals_not_shaped(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@below {{'kernel.linear_transform' op attribute 'diagonals' failed to satisfy constraint: constant vector/tensor attribute}}
  %0 = "kernel.linear_transform"(%arg0) {
    diagonals = 1 : i32,
    diagonal_indices = array<i64: 0>
  } : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @test_diagonals_not_2d(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@below {{diagonals must be a 2D tensor}}
  %0 = kernel.linear_transform %arg0 {
    diagonals = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>,
    diagonal_indices = array<i64: 0>
  } : tensor<4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @test_input_not_1d_or_2d(%arg0: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
  // expected-error@below {{input must be 1D or 2D ranked tensor}}
  %0 = kernel.linear_transform %arg0 {
    diagonals = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>,
    diagonal_indices = array<i64: 0, 1>
  } : tensor<1x2x3xf32> -> tensor<1x2x3xf32>
  return %0 : tensor<1x2x3xf32>
}

// -----

func.func @test_slot_size_mismatch(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@below {{input slot size (4) must match diagonals slot size (3)}}
  %0 = kernel.linear_transform %arg0 {
    diagonals = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>,
    diagonal_indices = array<i64: 0, 1>
  } : tensor<4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @test_diagonals_indices_mismatch(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@below {{number of diagonals (2) must match number of diagonal indices (1)}}
  %0 = kernel.linear_transform %arg0 {
    diagonals = dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>,
    diagonal_indices = array<i64: 0>
  } : tensor<4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @test_batch_dim_not_1(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  // expected-error@below {{input tensor batch dimension (first dimension) must be 1}}
  %0 = kernel.linear_transform %arg0 {
    diagonals = dense<[[1.0, 2.0, 3.0, 4.0]]> : tensor<1x4xf32>,
    diagonal_indices = array<i64: 0>
  } : tensor<2x4xf32> -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

