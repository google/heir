// RUN: heir-opt --verify-diagnostics --split-input-file %s

func.func @test_rotate_verifier_ok(%0: tensor<1x16xi32>) {
  %c1 = arith.constant 1 : i32
  %1 = tensor_ext.rotate %0, %c1 : tensor<1x16xi32>, i32
  return
}

// -----

func.func @rotate_reduce_rank(%0: tensor<16xi32>, %1: tensor<17x16xi32>) -> tensor<16xi32> {
  // expected-error@+1 {{requires plaintext tensor to have the same number of elements as steps}}
  %2 = tensor_ext.rotate_and_reduce %0, %1 {period = 1 : index, steps = 16 : index} : (tensor<16xi32>, tensor<17x16xi32>) -> tensor<16xi32>
  return %2 : tensor<16xi32>
}

// -----

func.func @rotate_rank(%0: tensor<16x16xi32>, %1: tensor<16x16xi32>) -> tensor<16x16xi32> {
  // expected-error@+1 {{requires a 1-D input tensor or tensor with single non-unit dimension}}
  %2 = tensor_ext.rotate_and_reduce %0, %1 {period = 1 : index, steps = 16 : index} : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi32>
  return %2 : tensor<16x16xi32>
}

// -----

func.func @rotate_reductions(%0: tensor<16xi32>, %1: tensor<32x16xi32>) -> tensor<16xi32> {
  // expected-error@+1 {{requires steps to be less than or equal to the input tensor's dimension}}
  %2 = tensor_ext.rotate_and_reduce %0, %1 {period = 1 : index, steps = 32 : index} : (tensor<16xi32>, tensor<32x16xi32>) -> tensor<16xi32>
  return %2 : tensor<16xi32>
}

// -----

func.func @rotate_reduce_period(%0: tensor<16xi32>, %1: tensor<16x16xi32>) -> tensor<16xi32> {
  // expected-error@+1 {{requires period to be within the range of the tensor}}
  %2 = tensor_ext.rotate_and_reduce %0, %1 {period = 22 : index, steps = 16 : index} : (tensor<16xi32>, tensor<16x16xi32>) -> tensor<16xi32>
  return %2 : tensor<16xi32>
}
