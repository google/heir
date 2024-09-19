// RUN: heir-opt --rotate-and-reduce --cse --canonicalize %s | FileCheck %s


// Sum all entries of a tensor into a single scalar
// CHECK-LABEL: @simple_sum
// CHECK-SAME: (%[[arg0:.*]]: tensor<8xi32>
// CHECK-NEXT: %[[c0:.*]] = arith.constant 0
// CHECK-NEXT: %[[c1:.*]] = arith.constant 1
// CHECK-NEXT: %[[c2:.*]] = arith.constant 2
// CHECK-NEXT: %[[c4:.*]] = arith.constant 4
// CHECK-NEXT: %[[v0:.*]] = tensor_ext.rotate %[[arg0]], %[[c4]]
// CHECK-NEXT: %[[v1:.*]] = arith.addi %[[arg0]], %[[v0]]
// CHECK-NEXT: %[[v2:.*]] = tensor_ext.rotate %[[v1]], %[[c2]]
// CHECK-NEXT: %[[v3:.*]] = arith.addi %[[v1]], %[[v2]]
// CHECK-NEXT: %[[v4:.*]] = tensor_ext.rotate %[[v3]], %[[c1]]
// CHECK-NEXT: %[[v5:.*]] = arith.addi %[[v3]], %[[v4]]
// CHECK-NEXT: %[[v6:.*]] = tensor.extract %[[v5]][%[[c0]]]
// CHECK-NEXT: return %[[v6]]
func.func @simple_sum(%arg0: tensor<8xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %0 = tensor.extract %arg0[%c0] : tensor<8xi32>
  %1 = tensor.extract %arg0[%c1] : tensor<8xi32>
  %2 = tensor.extract %arg0[%c2] : tensor<8xi32>
  %3 = tensor.extract %arg0[%c3] : tensor<8xi32>
  %4 = tensor.extract %arg0[%c4] : tensor<8xi32>
  %5 = tensor.extract %arg0[%c5] : tensor<8xi32>
  %6 = tensor.extract %arg0[%c6] : tensor<8xi32>
  %7 = tensor.extract %arg0[%c7] : tensor<8xi32>
  %8 = arith.addi %0, %1 : i32
  %9 = arith.addi %8, %2 : i32
  %10 = arith.addi %9, %3 : i32
  %11 = arith.addi %10, %4 : i32
  %12 = arith.addi %11, %5 : i32
  %13 = arith.addi %12, %6 : i32
  %14 = arith.addi %13, %7 : i32
  return %14 : i32
}

// Sum all entries of a tensor
// CHECK-LABEL: @simple_sum_mixed_rotation_tensor
// CHECK-SAME: (%[[arg0:.*]]: tensor<8xi32>
// CHECK-NEXT: %[[c1:.*]] = arith.constant 1
// CHECK-NEXT: %[[c2:.*]] = arith.constant 2
// CHECK-NEXT: %[[c4:.*]] = arith.constant 4
// CHECK-NEXT: %[[v0:.*]] = tensor_ext.rotate %[[arg0]], %[[c4]]
// CHECK-NEXT: %[[v1:.*]] = arith.addi %[[arg0]], %[[v0]]
// CHECK-NEXT: %[[v2:.*]] = tensor_ext.rotate %[[v1]], %[[c2]]
// CHECK-NEXT: %[[v3:.*]] = arith.addi %[[v1]], %[[v2]]
// CHECK-NEXT: %[[v4:.*]] = tensor_ext.rotate %[[v3]], %[[c1]]
// CHECK-NEXT: %[[v5:.*]] = arith.addi %[[v3]], %[[v4]]
// CHECK-NEXT: return %[[v5]]
func.func @simple_sum_mixed_rotation_tensor(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %0 = tensor_ext.rotate %arg0, %c1 : tensor<8xi32>, index
  %1 = tensor_ext.rotate %arg0, %c2 : tensor<8xi32>, index
  %2 = arith.addi %0, %1 : tensor<8xi32>
  %3 = tensor_ext.rotate %arg0, %c3 : tensor<8xi32>, index
  %4 = tensor_ext.rotate %arg0, %c4 : tensor<8xi32>, index
  %5 = tensor_ext.rotate %arg0, %c5 : tensor<8xi32>, index
  %6 = tensor_ext.rotate %arg0, %c6 : tensor<8xi32>, index
  %7 = tensor_ext.rotate %arg0, %c7 : tensor<8xi32>, index
  %8 = arith.addi %2, %3 : tensor<8xi32>
  %9 = arith.addi %8, %4 : tensor<8xi32>
  %10 = arith.addi %9, %5 : tensor<8xi32>
  %11 = arith.addi %10, %6 : tensor<8xi32>
  %12 = arith.addi %11, %7 : tensor<8xi32>
  %13 = arith.addi %12, %arg0 : tensor<8xi32>
  return %13 : tensor<8xi32>
}

// Sum all entries of a tensor into a single scalar
// Mix rotation and extraction in the reduction tree
// CHECK-LABEL: @simple_sum_mixed_rotation_extraction
// CHECK-SAME: (%[[arg0:.*]]: tensor<8xi32>
// CHECK-NEXT: %[[c0:.*]] = arith.constant 0
// CHECK-NEXT: %[[c1:.*]] = arith.constant 1
// CHECK-NEXT: %[[c2:.*]] = arith.constant 2
// CHECK-NEXT: %[[c4:.*]] = arith.constant 4
// CHECK-NEXT: %[[v0:.*]] = tensor_ext.rotate %[[arg0]], %[[c4]]
// CHECK-NEXT: %[[v1:.*]] = arith.addi %[[arg0]], %[[v0]]
// CHECK-NEXT: %[[v2:.*]] = tensor_ext.rotate %[[v1]], %[[c2]]
// CHECK-NEXT: %[[v3:.*]] = arith.addi %[[v1]], %[[v2]]
// CHECK-NEXT: %[[v4:.*]] = tensor_ext.rotate %[[v3]], %[[c1]]
// CHECK-NEXT: %[[v5:.*]] = arith.addi %[[v3]], %[[v4]]
// CHECK-NEXT: %[[v6:.*]] = tensor.extract %[[v5]][%[[c0]]]
// CHECK-NEXT: return %[[v6]]
func.func @simple_sum_mixed_rotation_extraction(%arg0: tensor<8xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %extracted = tensor.extract %arg0[%c0] : tensor<8xi32>
  %0 = tensor_ext.rotate %arg0, %c1 : tensor<8xi32>, index
  %1 = tensor_ext.rotate %arg0, %c2 : tensor<8xi32>, index
  %2 = arith.addi %0, %1 : tensor<8xi32>
  %extracted_0 = tensor.extract %2[%c0] : tensor<8xi32>
  %extracted_1 = tensor.extract %arg0[%c3] : tensor<8xi32>
  %extracted_2 = tensor.extract %arg0[%c4] : tensor<8xi32>
  %extracted_3 = tensor.extract %arg0[%c5] : tensor<8xi32>
  %extracted_4 = tensor.extract %arg0[%c6] : tensor<8xi32>
  %extracted_5 = tensor.extract %arg0[%c7] : tensor<8xi32>
  %3 = arith.addi %extracted, %extracted_0 : i32
  %4 = arith.addi %3, %extracted_1 : i32
  %5 = arith.addi %4, %extracted_2 : i32
  %6 = arith.addi %5, %extracted_3 : i32
  %7 = arith.addi %6, %extracted_4 : i32
  %8 = arith.addi %7, %extracted_5 : i32
  return %8 : i32
}

// CHECK-LABEL: @not_supported_mixed_ops
// CHECK-NOT: tensor_ext.rotate
func.func @not_supported_mixed_ops(%arg0: tensor<8xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %0 = tensor.extract %arg0[%c0] : tensor<8xi32>
  %1 = tensor.extract %arg0[%c1] : tensor<8xi32>
  %2 = tensor.extract %arg0[%c2] : tensor<8xi32>
  %3 = tensor.extract %arg0[%c3] : tensor<8xi32>
  %4 = tensor.extract %arg0[%c4] : tensor<8xi32>
  %5 = tensor.extract %arg0[%c5] : tensor<8xi32>
  %6 = tensor.extract %arg0[%c6] : tensor<8xi32>
  %7 = tensor.extract %arg0[%c7] : tensor<8xi32>
  %8 = arith.addi %0, %1 : i32
  %9 = arith.addi %8, %2 : i32
  %10 = arith.addi %9, %3 : i32
  %11 = arith.muli %10, %4 : i32
  %12 = arith.addi %11, %5 : i32
  %13 = arith.addi %12, %6 : i32
  %14 = arith.addi %13, %7 : i32
  return %14 : i32
}

// CHECK-LABEL: @not_supported_mixed_ops_mixed_rotation_extraction
// CHECK-COUNT-2: tensor_ext.rotate
// CHECK-NOT: tensor_ext.rotate
func.func @not_supported_mixed_ops_mixed_rotation_extraction(%arg0: tensor<8xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %extracted = tensor.extract %arg0[%c0] : tensor<8xi32>
  %0 = tensor_ext.rotate %arg0, %c1 : tensor<8xi32>, index
  %1 = tensor_ext.rotate %arg0, %c2 : tensor<8xi32>, index
  %2 = arith.muli %0, %1 : tensor<8xi32>
  %extracted_0 = tensor.extract %2[%c0] : tensor<8xi32>
  %extracted_1 = tensor.extract %arg0[%c3] : tensor<8xi32>
  %extracted_2 = tensor.extract %arg0[%c4] : tensor<8xi32>
  %extracted_3 = tensor.extract %arg0[%c5] : tensor<8xi32>
  %extracted_4 = tensor.extract %arg0[%c6] : tensor<8xi32>
  %extracted_5 = tensor.extract %arg0[%c7] : tensor<8xi32>
  %3 = arith.addi %extracted, %extracted_0 : i32
  %4 = arith.addi %3, %extracted_1 : i32
  %5 = arith.addi %4, %extracted_2 : i32
  %6 = arith.addi %5, %extracted_3 : i32
  %7 = arith.addi %6, %extracted_4 : i32
  %8 = arith.addi %7, %extracted_5 : i32
  return %8 : i32
}

// CHECK-LABEL: @not_supported_missing_indices
// CHECK-NOT: tensor_ext.rotate
func.func @not_supported_missing_indices(%arg0: tensor<16xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index
  %c12 = arith.constant 12 : index
  %c13 = arith.constant 13 : index
  %c14 = arith.constant 14 : index
  %0 = tensor.extract %arg0[%c0] : tensor<16xi32>
  %1 = tensor.extract %arg0[%c1] : tensor<16xi32>
  %2 = tensor.extract %arg0[%c2] : tensor<16xi32>
  %3 = tensor.extract %arg0[%c3] : tensor<16xi32>
  %4 = tensor.extract %arg0[%c4] : tensor<16xi32>
  %5 = tensor.extract %arg0[%c5] : tensor<16xi32>
  %6 = tensor.extract %arg0[%c6] : tensor<16xi32>
  %7 = tensor.extract %arg0[%c7] : tensor<16xi32>
  %8 = tensor.extract %arg0[%c8] : tensor<16xi32>
  %9 = tensor.extract %arg0[%c9] : tensor<16xi32>
  %10 = tensor.extract %arg0[%c10] : tensor<16xi32>
  %11 = tensor.extract %arg0[%c11] : tensor<16xi32>
  %12 = tensor.extract %arg0[%c12] : tensor<16xi32>
  %13 = tensor.extract %arg0[%c13] : tensor<16xi32>
  %14 = tensor.extract %arg0[%c14] : tensor<16xi32>
  // missing element 15
  %v1 = arith.addi %0, %1 : i32
  %v2 = arith.addi %v1, %2 : i32
  %v3 = arith.addi %v2, %3 : i32
  %v4 = arith.addi %v3, %4 : i32
  %v5 = arith.addi %v4, %5 : i32
  %v6 = arith.addi %v5, %6 : i32
  %v7 = arith.addi %v6, %7 : i32
  %v8 = arith.addi %v7, %8 : i32
  %v9 = arith.addi %v8, %9 : i32
  %v10 = arith.addi %v9, %10 : i32
  %v11 = arith.addi %v10, %11 : i32
  %v12 = arith.addi %v11, %12 : i32
  %v13 = arith.addi %v12, %13 : i32
  %v14 = arith.addi %v13, %14 : i32
  return %v14 : i32
}

// CHECK-LABEL: @not_supported_repeated_indices
// CHECK-NOT: tensor_ext.rotate
func.func @not_supported_repeated_indices(%arg0: tensor<8xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %0 = tensor.extract %arg0[%c0] : tensor<8xi32>
  %1 = tensor.extract %arg0[%c1] : tensor<8xi32>
  %2 = tensor.extract %arg0[%c2] : tensor<8xi32>
  %3 = tensor.extract %arg0[%c3] : tensor<8xi32>
  // repeats element 3
  %4 = tensor.extract %arg0[%c3] : tensor<8xi32>
  %5 = tensor.extract %arg0[%c5] : tensor<8xi32>
  %6 = tensor.extract %arg0[%c6] : tensor<8xi32>
  %7 = tensor.extract %arg0[%c7] : tensor<8xi32>
  %8 = arith.addi %0, %1 : i32
  %9 = arith.addi %8, %2 : i32
  %10 = arith.addi %9, %3 : i32
  %11 = arith.addi %10, %4 : i32
  %12 = arith.addi %11, %5 : i32
  %13 = arith.addi %12, %6 : i32
  %14 = arith.addi %13, %7 : i32
  return %14 : i32
}

// CHECK-LABEL: @not_supported_unsupported_op
// CHECK-NOT: tensor_ext.rotate
func.func @not_supported_unsupported_op(%arg0: tensor<8xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %0 = tensor.extract %arg0[%c0] : tensor<8xi32>
  %1 = tensor.extract %arg0[%c1] : tensor<8xi32>
  %2 = tensor.extract %arg0[%c2] : tensor<8xi32>
  %3 = tensor.extract %arg0[%c3] : tensor<8xi32>
  %4 = tensor.extract %arg0[%c4] : tensor<8xi32>
  %5 = tensor.extract %arg0[%c5] : tensor<8xi32>
  %6 = tensor.extract %arg0[%c6] : tensor<8xi32>
  %7 = tensor.extract %arg0[%c7] : tensor<8xi32>
  %8 = arith.subi %0, %1 : i32
  %9 = arith.subi %8, %2 : i32
  %10 = arith.subi %9, %3 : i32
  %11 = arith.subi %10, %4 : i32
  %12 = arith.subi %11, %5 : i32
  %13 = arith.subi %12, %6 : i32
  %14 = arith.subi %13, %7 : i32
  return %14 : i32
}

// 2D tensor not supported
// CHECK-LABEL: @not_supported_bad_tensor_shape
// CHECK-NOT: tensor_ext.rotate
func.func @not_supported_bad_tensor_shape(%arg0: tensor<1x8xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %0 = tensor.extract %arg0[%c1, %c0] : tensor<1x8xi32>
  %1 = tensor.extract %arg0[%c1, %c1] : tensor<1x8xi32>
  %2 = tensor.extract %arg0[%c1, %c2] : tensor<1x8xi32>
  %3 = tensor.extract %arg0[%c1, %c3] : tensor<1x8xi32>
  %4 = tensor.extract %arg0[%c1, %c4] : tensor<1x8xi32>
  %5 = tensor.extract %arg0[%c1, %c5] : tensor<1x8xi32>
  %6 = tensor.extract %arg0[%c1, %c6] : tensor<1x8xi32>
  %7 = tensor.extract %arg0[%c1, %c7] : tensor<1x8xi32>
  %8 = arith.addi %0, %1 : i32
  %9 = arith.addi %8, %2 : i32
  %10 = arith.addi %9, %3 : i32
  %11 = arith.addi %10, %4 : i32
  %12 = arith.addi %11, %5 : i32
  %13 = arith.addi %12, %6 : i32
  %14 = arith.addi %13, %7 : i32
  return %14 : i32
}

// reducing from multiple input tensors
// CHECK-LABEL: @not_supported_multiple_tensors
// CHECK-NOT: tensor_ext.rotate
func.func @not_supported_multiple_tensors(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %0 = tensor.extract %arg0[%c0] : tensor<8xi32>
  %1 = tensor.extract %arg0[%c1] : tensor<8xi32>
  // uses %arg1
  %2 = tensor.extract %arg1[%c2] : tensor<8xi32>
  %3 = tensor.extract %arg0[%c3] : tensor<8xi32>
  %4 = tensor.extract %arg0[%c4] : tensor<8xi32>
  %5 = tensor.extract %arg0[%c5] : tensor<8xi32>
  %6 = tensor.extract %arg0[%c6] : tensor<8xi32>
  %7 = tensor.extract %arg0[%c7] : tensor<8xi32>
  %8 = arith.addi %0, %1 : i32
  %9 = arith.addi %8, %2 : i32
  %10 = arith.addi %9, %3 : i32
  %11 = arith.addi %10, %4 : i32
  %12 = arith.addi %11, %5 : i32
  %13 = arith.addi %12, %6 : i32
  %14 = arith.addi %13, %7 : i32
  return %14 : i32
}

// CHECK-LABEL: @not_supported_non_constant_index_access
// CHECK-NOT: tensor_ext.rotate
func.func @not_supported_non_constant_index_access(%arg0: tensor<8xi32>, %arg1: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %0 = tensor.extract %arg0[%c0] : tensor<8xi32>
  %1 = tensor.extract %arg0[%c1] : tensor<8xi32>
  // uses non-constant index
  %2 = tensor.extract %arg0[%arg1] : tensor<8xi32>
  %3 = tensor.extract %arg0[%c3] : tensor<8xi32>
  %4 = tensor.extract %arg0[%c4] : tensor<8xi32>
  %5 = tensor.extract %arg0[%c5] : tensor<8xi32>
  %6 = tensor.extract %arg0[%c6] : tensor<8xi32>
  %7 = tensor.extract %arg0[%c7] : tensor<8xi32>
  %8 = arith.addi %0, %1 : i32
  %9 = arith.addi %8, %2 : i32
  %10 = arith.addi %9, %3 : i32
  %11 = arith.addi %10, %4 : i32
  %12 = arith.addi %11, %5 : i32
  %13 = arith.addi %12, %6 : i32
  %14 = arith.addi %13, %7 : i32
  return %14 : i32
}

// CHECK-LABEL: @not_supported_non_tensor_operands
// CHECK-NOT: tensor_ext.rotate
// TODO(#522): support this
func.func @not_supported_non_tensor_operands(%arg0: tensor<8xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c2_i32 = arith.constant 2 : i32
  %0 = tensor.extract %arg0[%c0] : tensor<8xi32>
  %1 = tensor.extract %arg0[%c1] : tensor<8xi32>
  %2 = tensor.extract %arg0[%c2] : tensor<8xi32>
  %3 = tensor.extract %arg0[%c3] : tensor<8xi32>
  %4 = tensor.extract %arg0[%c4] : tensor<8xi32>
  %5 = tensor.extract %arg0[%c5] : tensor<8xi32>
  %6 = tensor.extract %arg0[%c6] : tensor<8xi32>
  %7 = tensor.extract %arg0[%c7] : tensor<8xi32>
  %8 = arith.addi %0, %1 : i32
  // next op uses non-tensor operand
  %9 = arith.addi %8, %c2_i32 : i32
  %10 = arith.addi %9, %3 : i32
  %11 = arith.addi %10, %4 : i32
  %12 = arith.addi %11, %5 : i32
  %13 = arith.addi %12, %6 : i32
  %14 = arith.addi %13, %7 : i32
  %15 = arith.addi %14, %2 : i32
  return %15 : i32
}

// CHECK-LABEL: @sum_of_linear_rotates
// CHECK-COUNT-5: tensor_ext.rotate
// CHECK-NOT: tensor_ext.rotate
func.func @sum_of_linear_rotates(%arg0: !secret.secret<tensor<32xi16>>) -> !secret.secret<i16> {
  %c30 = arith.constant 30 : index
  %c29 = arith.constant 29 : index
  %c31 = arith.constant 31 : index
  %c1 = arith.constant 1 : index
  %0 = secret.generic ins(%arg0 : !secret.secret<tensor<32xi16>>) {
  ^bb0(%arg1: tensor<32xi16>):
    %1 = tensor_ext.rotate %arg1, %c1 : tensor<32xi16>, index
    %2 = arith.addi %1, %arg1 : tensor<32xi16>
    %3 = tensor_ext.rotate %arg1, %c31 : tensor<32xi16>, index
    %4 = tensor_ext.rotate %2, %c29 : tensor<32xi16>, index
    %5 = arith.addi %3, %4 : tensor<32xi16>
    %6 = arith.addi %5, %arg1 : tensor<32xi16>
    %7 = tensor_ext.rotate %6, %c30 : tensor<32xi16>, index
    %8 = arith.addi %3, %7 : tensor<32xi16>
    %9 = arith.addi %8, %arg1 : tensor<32xi16>
    %10 = tensor_ext.rotate %9, %c30 : tensor<32xi16>, index
    %11 = arith.addi %3, %10 : tensor<32xi16>
    %12 = arith.addi %11, %arg1 : tensor<32xi16>
    %13 = tensor_ext.rotate %12, %c30 : tensor<32xi16>, index
    %14 = arith.addi %3, %13 : tensor<32xi16>
    %15 = arith.addi %14, %arg1 : tensor<32xi16>
    %16 = tensor_ext.rotate %15, %c30 : tensor<32xi16>, index
    %17 = arith.addi %3, %16 : tensor<32xi16>
    %18 = arith.addi %17, %arg1 : tensor<32xi16>
    %19 = tensor_ext.rotate %18, %c30 : tensor<32xi16>, index
    %20 = arith.addi %3, %19 : tensor<32xi16>
    %21 = arith.addi %20, %arg1 : tensor<32xi16>
    %22 = tensor_ext.rotate %21, %c30 : tensor<32xi16>, index
    %23 = arith.addi %3, %22 : tensor<32xi16>
    %24 = arith.addi %23, %arg1 : tensor<32xi16>
    %25 = tensor_ext.rotate %24, %c30 : tensor<32xi16>, index
    %26 = arith.addi %3, %25 : tensor<32xi16>
    %27 = arith.addi %26, %arg1 : tensor<32xi16>
    %28 = tensor_ext.rotate %27, %c30 : tensor<32xi16>, index
    %29 = arith.addi %3, %28 : tensor<32xi16>
    %30 = arith.addi %29, %arg1 : tensor<32xi16>
    %31 = tensor_ext.rotate %30, %c30 : tensor<32xi16>, index
    %32 = arith.addi %3, %31 : tensor<32xi16>
    %33 = arith.addi %32, %arg1 : tensor<32xi16>
    %34 = tensor_ext.rotate %33, %c30 : tensor<32xi16>, index
    %35 = arith.addi %3, %34 : tensor<32xi16>
    %36 = arith.addi %35, %arg1 : tensor<32xi16>
    %37 = tensor_ext.rotate %36, %c30 : tensor<32xi16>, index
    %38 = arith.addi %3, %37 : tensor<32xi16>
    %39 = arith.addi %38, %arg1 : tensor<32xi16>
    %40 = tensor_ext.rotate %39, %c30 : tensor<32xi16>, index
    %41 = arith.addi %3, %40 : tensor<32xi16>
    %42 = arith.addi %41, %arg1 : tensor<32xi16>
    %43 = tensor_ext.rotate %42, %c30 : tensor<32xi16>, index
    %44 = arith.addi %3, %43 : tensor<32xi16>
    %45 = arith.addi %44, %arg1 : tensor<32xi16>
    %46 = tensor_ext.rotate %45, %c30 : tensor<32xi16>, index
    %47 = arith.addi %3, %46 : tensor<32xi16>
    %48 = arith.addi %47, %arg1 : tensor<32xi16>
    %extracted = tensor.extract %48[%c31] : tensor<32xi16>
    secret.yield %extracted : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}

// CHECK-LABEL: @rotate_not_applied_because_rotation_missing
// CHECK-COUNT-3: tensor_ext.rotate
func.func @rotate_not_applied_because_rotation_missing(%arg0: !secret.secret<tensor<4xi16>>) -> !secret.secret<i16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = secret.generic ins(%arg0 : !secret.secret<tensor<4xi16>>) {
  ^bb0(%arg1: tensor<4xi16>):
    %1 = tensor_ext.rotate %arg1, %c1 : tensor<4xi16>, index
    %2 = arith.addi %1, %arg1 : tensor<4xi16>
    %3 = tensor_ext.rotate %1, %c1 : tensor<4xi16>, index
    %4 = arith.addi %2, %3 : tensor<4xi16>
    // To make the rotation apply, replace %5 with this line
    // %5 = tensor_ext.rotate %3, %c1 : tensor<4xi16>, index
    %5 = tensor_ext.rotate %3, %c2 : tensor<4xi16>, index
    %6 = arith.addi %4, %5 : tensor<4xi16>
    %extracted = tensor.extract %6[%c0] : tensor<4xi16>
    secret.yield %extracted : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}

// CHECK-LABEL: @rotate_not_applied_because_rotation_duplicated
// CHECK-COUNT-3: tensor_ext.rotate
func.func @rotate_not_applied_because_rotation_duplicated(%arg0: !secret.secret<tensor<4xi16>>) -> !secret.secret<i16> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = secret.generic ins(%arg0 : !secret.secret<tensor<4xi16>>) {
  ^bb0(%arg1: tensor<4xi16>):
    %1 = tensor_ext.rotate %arg1, %c1 : tensor<4xi16>, index
    %2 = arith.addi %1, %arg1 : tensor<4xi16>
    %3 = tensor_ext.rotate %1, %c1 : tensor<4xi16>, index
    %4 = arith.addi %2, %3 : tensor<4xi16>
    // To return to normal, replace %v4_2 with %4
    %v4_2 = arith.addi %4, %3 : tensor<4xi16>
    %5 = tensor_ext.rotate %3, %c1 : tensor<4xi16>, index
    %6 = arith.addi %v4_2, %5 : tensor<4xi16>
    %extracted = tensor.extract %6[%c1] : tensor<4xi16>
    secret.yield %extracted : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}

// CHECK-LABEL: @rotate_not_applied_because_multiple_tensors
// CHECK-COUNT-3: tensor_ext.rotate
func.func @rotate_not_applied_because_multiple_tensors(
    %arg0 : tensor<4xi16>, %arg1 : tensor<4xi16>) -> i16 {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %1 = tensor_ext.rotate %arg1, %c1 : tensor<4xi16>, index
  %2 = arith.addi %1, %arg1 : tensor<4xi16>
  %3 = tensor_ext.rotate %1, %c1 : tensor<4xi16>, index
  %4 = arith.addi %2, %3 : tensor<4xi16>
  // To return to normal, replace %v4_2 with %4
  %v4_2 = arith.addi %4, %arg0 : tensor<4xi16>
  %5 = tensor_ext.rotate %3, %c1 : tensor<4xi16>, index
  %6 = arith.addi %v4_2, %5 : tensor<4xi16>
  %extracted = tensor.extract %6[%c1] : tensor<4xi16>
  return %extracted : i16
}

// CHECK-LABEL: @rotate_not_applied_because_mixed_ops
// CHECK-COUNT-3: tensor_ext.rotate
func.func @rotate_not_applied_because_mixed_ops(%arg1 : tensor<4xi16>) -> i16 {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %1 = tensor_ext.rotate %arg1, %c1 : tensor<4xi16>, index
  %2 = arith.addi %1, %arg1 : tensor<4xi16>
  %3 = tensor_ext.rotate %1, %c1 : tensor<4xi16>, index
  // To return to normal, replace muli with addi
  %4 = arith.muli %2, %3 : tensor<4xi16>
  %5 = tensor_ext.rotate %3, %c1 : tensor<4xi16>, index
  %6 = arith.addi %4, %5 : tensor<4xi16>
  %extracted = tensor.extract %6[%c1] : tensor<4xi16>
  return %extracted : i16
}

// CHECK-LABEL: @reduce_add_and_mul
// 9 rotations because the first rotation can be re-used between the two
// reductions
// CHECK-COUNT-9: tensor_ext.rotate
// CHECK-NOT: tensor_ext.rotate
func.func @reduce_add_and_mul(%arg1: tensor<32xi16>) -> i16 {
  %c30 = arith.constant 30 : index
  %c29 = arith.constant 29 : index
  %c31 = arith.constant 31 : index
  %c1 = arith.constant 1 : index

  // the add reduction
  %1 = tensor_ext.rotate %arg1, %c1 : tensor<32xi16>, index
  %2 = arith.addi %1, %arg1 : tensor<32xi16>
  %3 = tensor_ext.rotate %arg1, %c31 : tensor<32xi16>, index
  %4 = tensor_ext.rotate %2, %c29 : tensor<32xi16>, index
  %5 = arith.addi %3, %4 : tensor<32xi16>
  %6 = arith.addi %5, %arg1 : tensor<32xi16>
  %7 = tensor_ext.rotate %6, %c30 : tensor<32xi16>, index
  %8 = arith.addi %3, %7 : tensor<32xi16>
  %9 = arith.addi %8, %arg1 : tensor<32xi16>
  %10 = tensor_ext.rotate %9, %c30 : tensor<32xi16>, index
  %11 = arith.addi %3, %10 : tensor<32xi16>
  %12 = arith.addi %11, %arg1 : tensor<32xi16>
  %13 = tensor_ext.rotate %12, %c30 : tensor<32xi16>, index
  %14 = arith.addi %3, %13 : tensor<32xi16>
  %15 = arith.addi %14, %arg1 : tensor<32xi16>
  %16 = tensor_ext.rotate %15, %c30 : tensor<32xi16>, index
  %17 = arith.addi %3, %16 : tensor<32xi16>
  %18 = arith.addi %17, %arg1 : tensor<32xi16>
  %19 = tensor_ext.rotate %18, %c30 : tensor<32xi16>, index
  %20 = arith.addi %3, %19 : tensor<32xi16>
  %21 = arith.addi %20, %arg1 : tensor<32xi16>
  %22 = tensor_ext.rotate %21, %c30 : tensor<32xi16>, index
  %23 = arith.addi %3, %22 : tensor<32xi16>
  %24 = arith.addi %23, %arg1 : tensor<32xi16>
  %25 = tensor_ext.rotate %24, %c30 : tensor<32xi16>, index
  %26 = arith.addi %3, %25 : tensor<32xi16>
  %27 = arith.addi %26, %arg1 : tensor<32xi16>
  %28 = tensor_ext.rotate %27, %c30 : tensor<32xi16>, index
  %29 = arith.addi %3, %28 : tensor<32xi16>
  %30 = arith.addi %29, %arg1 : tensor<32xi16>
  %31 = tensor_ext.rotate %30, %c30 : tensor<32xi16>, index
  %32 = arith.addi %3, %31 : tensor<32xi16>
  %33 = arith.addi %32, %arg1 : tensor<32xi16>
  %34 = tensor_ext.rotate %33, %c30 : tensor<32xi16>, index
  %35 = arith.addi %3, %34 : tensor<32xi16>
  %36 = arith.addi %35, %arg1 : tensor<32xi16>
  %37 = tensor_ext.rotate %36, %c30 : tensor<32xi16>, index
  %38 = arith.addi %3, %37 : tensor<32xi16>
  %39 = arith.addi %38, %arg1 : tensor<32xi16>
  %40 = tensor_ext.rotate %39, %c30 : tensor<32xi16>, index
  %41 = arith.addi %3, %40 : tensor<32xi16>
  %42 = arith.addi %41, %arg1 : tensor<32xi16>
  %43 = tensor_ext.rotate %42, %c30 : tensor<32xi16>, index
  %44 = arith.addi %3, %43 : tensor<32xi16>
  %45 = arith.addi %44, %arg1 : tensor<32xi16>
  %46 = tensor_ext.rotate %45, %c30 : tensor<32xi16>, index
  %47 = arith.addi %3, %46 : tensor<32xi16>
  %48 = arith.addi %47, %arg1 : tensor<32xi16>
  %extracted = tensor.extract %48[%c31] : tensor<32xi16>

  // the mul reduction
  %v1_2 = tensor_ext.rotate %arg1, %c1 : tensor<32xi16>, index
  %v2_2 = arith.muli %v1_2, %arg1 : tensor<32xi16>
  %v3_2 = tensor_ext.rotate %arg1, %c31 : tensor<32xi16>, index
  %v4_2 = tensor_ext.rotate %v2_2, %c29 : tensor<32xi16>, index
  %v5_2 = arith.muli %v3_2, %v4_2 : tensor<32xi16>
  %v6_2 = arith.muli %v5_2, %arg1 : tensor<32xi16>
  %v7_2 = tensor_ext.rotate %v6_2, %c30 : tensor<32xi16>, index
  %v8_2 = arith.muli %v3_2, %v7_2 : tensor<32xi16>
  %v9_2 = arith.muli %v8_2, %arg1 : tensor<32xi16>
  %v10_2 = tensor_ext.rotate %v9_2, %c30 : tensor<32xi16>, index
  %v11_2 = arith.muli %v3_2, %v10_2 : tensor<32xi16>
  %v12_2 = arith.muli %v11_2, %arg1 : tensor<32xi16>
  %v13_2 = tensor_ext.rotate %v12_2, %c30 : tensor<32xi16>, index
  %v14_2 = arith.muli %v3_2, %v13_2 : tensor<32xi16>
  %v15_2 = arith.muli %v14_2, %arg1 : tensor<32xi16>
  %v16_2 = tensor_ext.rotate %v15_2, %c30 : tensor<32xi16>, index
  %v17_2 = arith.muli %v3_2, %v16_2 : tensor<32xi16>
  %v18_2 = arith.muli %v17_2, %arg1 : tensor<32xi16>
  %v19_2 = tensor_ext.rotate %v18_2, %c30 : tensor<32xi16>, index
  %v20_2 = arith.muli %v3_2, %v19_2 : tensor<32xi16>
  %v21_2 = arith.muli %v20_2, %arg1 : tensor<32xi16>
  %v22_2 = tensor_ext.rotate %v21_2, %c30 : tensor<32xi16>, index
  %v23_2 = arith.muli %v3_2, %v22_2 : tensor<32xi16>
  %v24_2 = arith.muli %v23_2, %arg1 : tensor<32xi16>
  %v25_2 = tensor_ext.rotate %v24_2, %c30 : tensor<32xi16>, index
  %v26_2 = arith.muli %v3_2, %v25_2 : tensor<32xi16>
  %v27_2 = arith.muli %v26_2, %arg1 : tensor<32xi16>
  %v28_2 = tensor_ext.rotate %v27_2, %c30 : tensor<32xi16>, index
  %v29_2 = arith.muli %v3_2, %v28_2 : tensor<32xi16>
  %v30_2 = arith.muli %v29_2, %arg1 : tensor<32xi16>
  %v31_2 = tensor_ext.rotate %v30_2, %c30 : tensor<32xi16>, index
  %v32_2 = arith.muli %v3_2, %v31_2 : tensor<32xi16>
  %v33_2 = arith.muli %v32_2, %arg1 : tensor<32xi16>
  %v34_2 = tensor_ext.rotate %v33_2, %c30 : tensor<32xi16>, index
  %v35_2 = arith.muli %v3_2, %v34_2 : tensor<32xi16>
  %v36_2 = arith.muli %v35_2, %arg1 : tensor<32xi16>
  %v37_2 = tensor_ext.rotate %v36_2, %c30 : tensor<32xi16>, index
  %v38_2 = arith.muli %v3_2, %v37_2 : tensor<32xi16>
  %v39_2 = arith.muli %v38_2, %arg1 : tensor<32xi16>
  %v40_2 = tensor_ext.rotate %v39_2, %c30 : tensor<32xi16>, index
  %v41_2 = arith.muli %v3_2, %v40_2 : tensor<32xi16>
  %v42_2 = arith.muli %v41_2, %arg1 : tensor<32xi16>
  %v43_2 = tensor_ext.rotate %v42_2, %c30 : tensor<32xi16>, index
  %v44_2 = arith.muli %v3_2, %v43_2 : tensor<32xi16>
  %v45_2 = arith.muli %v44_2, %arg1 : tensor<32xi16>
  %v46_2 = tensor_ext.rotate %v45_2, %c30 : tensor<32xi16>, index
  %v47_2 = arith.muli %v3_2, %v46_2 : tensor<32xi16>
  %v48_2 = arith.muli %v47_2, %arg1 : tensor<32xi16>
  %extracted_2 = tensor.extract %v48_2[%c31] : tensor<32xi16>

  %out = arith.addi %extracted, %extracted_2 : i16
  return %out : i16
}


// This test caused rotate-and-reduce to crash, so is here as a regression test
// without any particular assertion required.
// CHECK-LABEL: @test_dot_product_regression
func.func @test_dot_product_regression(%arg0: !secret.secret<tensor<8xi16>>, %arg1: !secret.secret<tensor<8xi16>>) -> !secret.secret<i16> {
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>) {
  ^bb0(%arg2: tensor<8xi16>, %arg3: tensor<8xi16>):
    %1 = arith.muli %arg2, %arg3 : tensor<8xi16>
    %2 = tensor_ext.rotate %1, %c1 : tensor<8xi16>, index
    %3 = arith.addi %2, %1 : tensor<8xi16>
    %4 = tensor_ext.rotate %1, %c2 : tensor<8xi16>, index
    %5 = arith.addi %4, %3 : tensor<8xi16>
    %6 = tensor_ext.rotate %1, %c3 : tensor<8xi16>, index
    %7 = arith.addi %6, %5 : tensor<8xi16>
    %extracted = tensor.extract %7[%c0] : tensor<8xi16>
    secret.yield %extracted : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
