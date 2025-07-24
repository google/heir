// RUN: heir-opt --baby-step-giant-step --canonicalize --split-input-file %s | FileCheck %s

// CHECK: @test_baby_step_giant_step_4
func.func @test_baby_step_giant_step_4(%0: tensor<16xi32> {secret.secret}, %1: tensor<16x16xi32>) -> tensor<16xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  // Extract rows of plaintext
  %row0 = tensor.extract_slice %1[0, 0][1, 16][1, 1] : tensor<16x16xi32> to tensor<16xi32>
  %row1 = tensor.extract_slice %1[1, 0][1, 16][1, 1] : tensor<16x16xi32> to tensor<16xi32>
  %row2 = tensor.extract_slice %1[2, 0][1, 16][1, 1] : tensor<16x16xi32> to tensor<16xi32>
  %row3 = tensor.extract_slice %1[3, 0][1, 16][1, 1] : tensor<16x16xi32> to tensor<16xi32>
  // Compute rotations of the plaintext
  %rot1 = tensor_ext.rotate %0, %c1 : tensor<16xi32>, index
  %rot2 = tensor_ext.rotate %0, %c2 : tensor<16xi32>, index
  %rot3 = tensor_ext.rotate %0, %c3 : tensor<16xi32>, index
  // Compute products
  %row0_rot0 = arith.muli %row0, %0 : tensor<16xi32>
  %row1_rot1 = arith.muli %row1, %rot1 : tensor<16xi32>
  %row2_rot2 = arith.muli %row2, %rot2 : tensor<16xi32>
  %row3_rot3 = arith.muli %row3, %rot3 : tensor<16xi32>
  // Add the rows together
  %sum0 = arith.addi %row0_rot0, %row1_rot1 : tensor<16xi32>
  %sum1 = arith.addi %sum0, %row2_rot2 : tensor<16xi32>
  %sum2 = arith.addi %sum1, %row3_rot3 : tensor<16xi32>
  return %sum2 : tensor<16xi32>
}

// -----

// Tests iterative rotations
// CHECK: @test_baby_step_giant_step_4_iterative
func.func @test_baby_step_giant_step_4_iterative(%0: tensor<16xi32> {secret.secret}, %1: tensor<16x16xi32>) -> tensor<16xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  // Extract rows of plaintext
  %row0 = tensor.extract_slice %1[0, 0][1, 16][1, 1] : tensor<16x16xi32> to tensor<16xi32>
  %row1 = tensor.extract_slice %1[1, 0][1, 16][1, 1] : tensor<16x16xi32> to tensor<16xi32>
  %row2 = tensor.extract_slice %1[2, 0][1, 16][1, 1] : tensor<16x16xi32> to tensor<16xi32>
  %row3 = tensor.extract_slice %1[3, 0][1, 16][1, 1] : tensor<16x16xi32> to tensor<16xi32>
  // Compute rotations of the plaintext
  %rot1 = tensor_ext.rotate %0, %c1 : tensor<16xi32>, index
  %rot2 = tensor_ext.rotate %rot1, %c1 : tensor<16xi32>, index
  %rot3 = tensor_ext.rotate %rot2, %c1 : tensor<16xi32>, index
  // Compute products
  %row0_rot0 = arith.muli %row0, %0 : tensor<16xi32>
  %row1_rot1 = arith.muli %row1, %rot1 : tensor<16xi32>
  %row2_rot2 = arith.muli %row2, %rot2 : tensor<16xi32>
  %row3_rot3 = arith.muli %row3, %rot3 : tensor<16xi32>
  // Add the rows together
  %sum0 = arith.addi %row0_rot0, %row1_rot1 : tensor<16xi32>
  %sum1 = arith.addi %sum0, %row2_rot2 : tensor<16xi32>
  %sum2 = arith.addi %sum1, %row3_rot3 : tensor<16xi32>
  return %sum2 : tensor<16xi32>
}

// -----

// Tests different reductions
// CHECK: @test_baby_step_giant_step_4_reductions
func.func @test_baby_step_giant_step_4_reductions(%0: tensor<16xi32> {secret.secret}, %1: tensor<16x16xi32>) -> tensor<16xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  // Extract rows of plaintext
  %row0 = tensor.extract_slice %1[0, 0][1, 16][1, 1] : tensor<16x16xi32> to tensor<16xi32>
  %row1 = tensor.extract_slice %1[1, 0][1, 16][1, 1] : tensor<16x16xi32> to tensor<16xi32>
  %row2 = tensor.extract_slice %1[2, 0][1, 16][1, 1] : tensor<16x16xi32> to tensor<16xi32>
  %row3 = tensor.extract_slice %1[3, 0][1, 16][1, 1] : tensor<16x16xi32> to tensor<16xi32>
  // Compute rotations of the plaintext
  %rot1 = tensor_ext.rotate %0, %c1 : tensor<16xi32>, index
  %rot2 = tensor_ext.rotate %0, %c2 : tensor<16xi32>, index
  %rot3 = tensor_ext.rotate %0, %c3 : tensor<16xi32>, index
  // Compute products
  %row0_rot0 = arith.muli %row0, %0 : tensor<16xi32>
  %row1_rot1 = arith.muli %row1, %rot1 : tensor<16xi32>
  %row2_rot2 = arith.muli %row2, %rot2 : tensor<16xi32>
  %row3_rot3 = arith.muli %row3, %rot3 : tensor<16xi32>
  // Add the rows together
  %sum0 = arith.addi %row0_rot0, %row1_rot1 : tensor<16xi32>
  %sum1 = arith.addi %row2_rot2, %row3_rot3 : tensor<16xi32>
  %sum2 = arith.addi %sum0, %sum1 : tensor<16xi32>
  return %sum2 : tensor<16xi32>
}
