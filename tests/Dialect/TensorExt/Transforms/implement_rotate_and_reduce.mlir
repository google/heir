// RUN: heir-opt --implement-rotate-and-reduce --canonicalize --cse --sccp %s | FileCheck %s

// CHECK: @test_halevi_shoup_reduction
// CHECK-SAME: %[[arg0:.*]]: tensor<16xi32>, %[[arg1:.*]]: tensor<16x16xi32>
// CHECK-DAG: %[[cneg4:.*]] = arith.constant -4 : i32
// CHECK-DAG: %[[cneg8:.*]] = arith.constant -8 : i32
// CHECK-DAG: %[[cneg12:.*]] = arith.constant -12 : i32
// CHECK-DAG: %[[c4:.*]] = arith.constant 4 : i32
// CHECK-DAG: %[[c8:.*]] = arith.constant 8 : i32
// CHECK-DAG: %[[c12:.*]] = arith.constant 12 : i32


// Giant steps (chunked in four) will also need 3 rotations to align the sums.
// Each giant step consists of 4 plaintext extractions. muls, and rotates

// Baby step giant step should reduce the number of ciphertext rotations to 3 (shifts of 0, 1, 2, 3)
// First chunk of 4 baby steps is interleaved with the initial rotations.
// CHECK: arith.muli
// CHECK: tensor_ext.rotate %[[arg0]]
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: tensor_ext.rotate %[[arg0]]
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: tensor_ext.rotate %[[arg0]]
// CHECK: arith.muli
// CHECK: arith.addi

// Plaintexts are all rotated by a fixed amount and multiplied by the baby step ciphertexts
// The result is shifted back to the correct position
// CHECK-COUNT-4: tensor_ext.rotate %{{.*}}, %[[cneg4]]
// CHECK: tensor_ext.rotate %{{.*}}, %[[c4]]

// CHECK-COUNT-4: tensor_ext.rotate %{{.*}}, %[[cneg8]]
// CHECK: tensor_ext.rotate %{{.*}}, %[[c8]]

// CHECK-COUNT-4: tensor_ext.rotate %{{.*}}, %[[cneg12]]
// CHECK: tensor_ext.rotate %{{.*}}, %[[c12]]
// CHECK: return

func.func @test_halevi_shoup_reduction(%0: tensor<16xi32>, %1: tensor<16x16xi32>) -> tensor<16xi32> {
  %2 = tensor_ext.rotate_and_reduce %0, %1 {period = 1 : index, steps = 16 : index} : (tensor<16xi32>, tensor<16x16xi32>) -> tensor<16xi32>
  return %2 : tensor<16xi32>
}
